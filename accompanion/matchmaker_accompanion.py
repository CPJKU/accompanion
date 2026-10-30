# -*- coding: utf-8 -*-
"""
Matchmaker ACCompanion.

Runs the ACCompanion with any score follower Matchmaker provides, selected by
name. Unlike `HMMACCompanion` and `OLTWACCompanion`, which wire up one
particular follower with the ACCompanion's own tuning, this class knows nothing
about the follower it builds: the method, its feature processor and its
arguments all come from Matchmaker's registry, so a method added to Matchmaker
later -- in ``methods.yaml`` or through `matchmaker.register_method` -- works
here without any change.

Run `python bin/test_followers.py --list` to see what is available.
"""
from os import PathLike
from typing import Any, Dict, Optional

from accompanion.hmm_accompanion import HMMACCompanion
from accompanion.score_follower.matchmaker_methods import (
    available_methods,
    build_score_follower,
    default_method,
)
from accompanion.score_follower.trackers import MatchmakerScoreFollower

SCORE_FOLLOWER_DEFAULT_KWARGS: Dict[str, Any] = {
    # Matchmaker's PitchIOIHMM, the closest relative of the ACCompanion's own
    # HMM follower.
    "score_follower": "hmm",
    "score_follower_kwargs": {},
}


class MatchmakerACCompanion(HMMACCompanion):
    """
    The ACCompanion running an arbitrary Matchmaker score follower.

    The score setup is inherited from `HMMACCompanion` -- a solo score, an
    accompaniment score and an optional accompaniment match file, all loaded
    with partitura. Only the score follower differs.

    `score_follower_kwargs` is read as:

    ``score_follower``
        The Matchmaker method name, e.g. ``"hmm"``, ``"pthmm"``, ``"arzt"``,
        ``"dixon"``, ``"outerhmm"``, ``"pfkorz"``, ``"OPTM"``. See
        `accompanion.score_follower.matchmaker_methods.available_methods`.
    ``score_follower_kwargs``
        Method configuration, merged over the method's own defaults from
        Matchmaker's spec. This is Matchmaker's ``kwargs`` dict, so the keys
        each method accepts are the ones it declares in ``methods.yaml``
        (``piano_range``, ``window_size``, ``step_size``, ``num_particles``,
        and so on).

    Notes
    -----
    Some methods are *event based*: they align note by note and reject a frame
    holding a chord. For those the MIDI input switches to handing the follower
    one message at a time, which is reported at startup.
    """

    def __init__(
        self,
        solo_fn: PathLike,
        acc_fn: PathLike,
        midi_router_kwargs: dict,
        score_follower_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            solo_fn=solo_fn,
            acc_fn=acc_fn,
            midi_router_kwargs=midi_router_kwargs,
            score_follower_kwargs=(
                dict(SCORE_FOLLOWER_DEFAULT_KWARGS)
                if score_follower_kwargs is None
                else score_follower_kwargs
            ),
            **kwargs,
        )

    def setup_score_follower(self):
        """
        Setup the score follower object.

        Everything method-specific is resolved by Matchmaker's registry; this
        method only supplies the ACCompanion's score, tempo and polling period.
        """
        # The input processor is the method's own, as declared in Matchmaker's
        # spec, so an `input_processor` entry in the config is not used here.
        self.score_follower_kwargs.pop("input_processor", None)
        method = self.score_follower_kwargs.pop("score_follower", None)
        if method is None:
            method = default_method()

        config = self.score_follower_kwargs.pop("score_follower_kwargs", None) or {}
        # Anything left at the top level is method configuration too, which is
        # how the OLTW configs already spell `window_size` and `step_size`.
        config = {**self.score_follower_kwargs, **config}

        if method not in available_methods():
            raise ValueError(
                f"'{method}' is not a Matchmaker score follower. Available: "
                f"{sorted(available_methods())}. The ACCompanion's own "
                "followers are 'PitchIOIHMM' and 'PitchIOIKHMM' "
                "(HMMACCompanion) and 'OnlineTimeWarping' (OLTWACCompanion)."
            )

        processor, follower, info = build_score_follower(
            method=method,
            score_part=self.solo_spart,
            score_positions=self.solo_score.unique_onsets,
            tempo=60 / self.init_bp,
            polling_period=self.polling_period,
            config=config,
        )

        # A method whose spec asks for a null polling period aligns note by
        # note and rejects a frame holding a chord, so the MIDI input has to
        # hand it one message at a time.
        self.event_based_input = info["event_based"]

        print(
            f"Score follower: matchmaker '{info['method']}' "
            f"(processor '{info['processor']}', "
            f"input {'event based' if self.event_based_input else 'framed'})"
        )

        self.score_follower = MatchmakerScoreFollower(follower)
        self.input_pipeline = processor

    def check_empty_frames(self, frame):
        """
        Check if the frame is empty.

        Parameters
        ----------
        frame : tuple or None
            The output of the input pipeline, i.e. a (features, performance
            time) tuple, or None.

        Returns
        -------
        bool
        """
        if frame is None:
            return True

        features, _ = frame
        # Note-array rows (the parangonar processor) are never empty.
        return bool(getattr(features, "dtype", None) is not None
                    and features.dtype.fields is None
                    and not features.any())
