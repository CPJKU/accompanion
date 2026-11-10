# -*- coding: utf-8 -*-
"""
Building any of Matchmaker's score followers for the ACCompanion.

Matchmaker declares its score followers in ``matchmaker/methods.yaml`` and
interprets that spec in :mod:`matchmaker.registry`; followers that live outside
the package are added at runtime with :func:`matchmaker.register_method`. Both
kinds are built from a :class:`matchmaker.Matchmaker` instance, off which the
spec's *providers* read whatever a given method needs -- the score part, the
reference features, the tempo, the stream queue, and so on.

The ACCompanion cannot use :class:`matchmaker.Matchmaker` itself: it owns its
MIDI I/O, its score objects and its main loop, and it feeds the follower
frame by frame instead of letting the follower pull from a stream.
:class:`FollowerContext` below stands in for the ``Matchmaker`` instance,
exposing the same attributes the providers read. Everything method-specific
then stays in Matchmaker's spec, so a method added to ``methods.yaml``
upstream, or registered at runtime with ``register_method``, becomes usable
here without any change to the ACCompanion.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matchmaker.features.audio import SAMPLE_RATE
from matchmaker.io.midi import POLLING_PERIOD as MIDI_STREAM_POLLING_PERIOD
from matchmaker.io.queue import RECVQueue
from matchmaker.matchmaker import (
    AVAILABLE_METHODS,
    CUSTOM_METHODS,
    DEFAULT_KWARGS,
    DEFAULT_METHOD,
    DEFAULT_PROCESSOR,
    MIDI_FRAME_RATE,
)
from matchmaker.registry import REGISTRY

#: The ACCompanion only ever follows a MIDI stream.
INPUT_TYPE = "midi"


def available_methods() -> List[str]:
    """Names of every Matchmaker method the ACCompanion can follow with.

    This is read live from Matchmaker, so it also lists methods registered
    with `matchmaker.register_method` and any method added to
    ``matchmaker/methods.yaml`` in a newer version of the package.
    """
    return list(AVAILABLE_METHODS.get(INPUT_TYPE, []))


def default_method() -> str:
    """Matchmaker's default MIDI method."""
    return DEFAULT_METHOD[INPUT_TYPE]


def method_defaults(method: str) -> Dict[str, Any]:
    """The method's ``default_kwargs``, as declared in Matchmaker's spec."""
    return dict(DEFAULT_KWARGS.get(INPUT_TYPE, {}).get(method, {}))


def _custom_spec(method: str) -> Optional[dict]:
    """The `register_method` registration for `method`, or None if built in."""
    return CUSTOM_METHODS.get((INPUT_TYPE, method))


def is_event_based(method: str) -> bool:
    """Whether the method is flagged ``event_based`` in Matchmaker's spec."""
    spec = REGISTRY.methods.get(INPUT_TYPE, {}).get(method)
    return bool(spec is not None and spec.event_based)


def preferred_polling_period(method: str) -> Optional[float]:
    """The polling period the method's spec asks for, in seconds.

    None means the method has to be fed one MIDI message at a time: it aligns
    note by note and rejects a frame holding a chord. Matchmaker's
    `Matchmaker` applies exactly this default when the user sets none.
    """
    defaults = method_defaults(method)
    if "polling_period" in defaults:
        return defaults["polling_period"]
    return None if is_event_based(method) else MIDI_STREAM_POLLING_PERIOD


class FollowerContext(object):
    """Stand-in for a `matchmaker.Matchmaker` instance.

    Carries exactly the attributes Matchmaker's argument providers and
    reference builders read (see `matchmaker.registry.PROVIDERS` and
    `REFERENCE_BUILDERS`). The ACCompanion supplies its own score and its own
    notion of tempo and polling period; everything else keeps Matchmaker's
    defaults.

    Parameters
    ----------
    method : str
        The Matchmaker method being built.
    score_part : partitura.score.Part
        The solo part the follower aligns against.
    score_positions : np.ndarray
        Ascending score onsets in beats. These are the ACCompanion's own
        `solo_score.unique_onsets`, so that the positions a follower reports
        line up with the onset tracker.
    polling_period : float or None
        Duration of one input frame, in seconds. None for event-based methods.
    tempo : float
        Tempo in BPM, used by the methods that need a notated tempo.
    config : dict
        Method configuration, i.e. Matchmaker's ``kwargs``. Consumed during
        the build: what is left over is splatted into the follower's
        constructor by methods declaring ``config_passthrough``.
    """

    input_type: str = INPUT_TYPE

    def __init__(
        self,
        method: str,
        score_part,
        score_positions: np.ndarray,
        polling_period: Optional[float],
        tempo: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.method = method
        self.score_part = score_part
        self._score_positions = np.asarray(score_positions)
        self.polling_period = polling_period
        self.tempo = float(tempo)
        self.config: Dict[str, Any] = dict(config or {})

        # MIDI input has no frame rate; Matchmaker uses a dummy value.
        self.frame_rate = MIDI_FRAME_RATE
        # Only read by methods that render the score as audio. Kept at
        # Matchmaker's defaults.
        self.sample_rate = SAMPLE_RATE
        self.hop_length = 1

        # The ACCompanion calls the follower directly rather than letting it
        # pull from a stream, so this queue stays empty. It exists because the
        # spec hands every follower a queue.
        self.stream = _DetachedStream()

        # Filled in by `build_score_follower` as the build progresses; the
        # providers read them back.
        self.processor = None
        self.reference_features = None
        self.performance_file = None

    @property
    def score_positions(self) -> np.ndarray:
        return self._score_positions

    def ref_frame_to_beat(self) -> np.ndarray:
        """Beat position of each reference frame.

        Only meaningful for followers aligning against a frame-wise rendering
        of the score, which for MIDI input none of them do.
        """
        raise NotImplementedError(
            f"Method '{self.method}' asks for a per-frame beat map of the "
            "reference, which the ACCompanion only builds for its own "
            "'OnlineTimeWarping' follower."
        )


class _DetachedStream(object):
    """A stream that only ever hands out its (unused) queue."""

    def __init__(self) -> None:
        self.queue = RECVQueue()


def build_score_follower(
    method: str,
    score_part,
    score_positions: np.ndarray,
    tempo: float,
    polling_period: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Build a Matchmaker feature processor and score follower.

    Mirrors the build order of `matchmaker.Matchmaker.__init__`: the processor
    first, then the score-side reference features, then the follower.

    Parameters
    ----------
    method : str
        One of `available_methods()`.
    score_part : partitura.score.Part
        The solo part to follow.
    score_positions : np.ndarray
        The ACCompanion's unique score onsets, in beats.
    tempo : float
        Tempo in BPM.
    polling_period : float or None
        The ACCompanion's polling period. If None, the method's preferred one
        is used.
    config : dict, optional
        Method configuration. Merged over the method's `default_kwargs`.

    Returns
    -------
    processor : matchmaker.features.processor.Processor
        The input pipeline for this method.
    follower : matchmaker.base.OnlineAlignment
        The score follower.
    info : dict
        What the build resolved: ``method``, ``processor`` (name),
        ``polling_period`` (what the follower was built with),
        ``wanted_polling_period`` (what its spec asks to be fed at) and
        ``event_based`` (whether it has to be fed one MIDI message at a time).
    """
    methods = available_methods()
    if method not in methods:
        raise ValueError(
            f"'{method}' is not a Matchmaker MIDI score follower. "
            f"Available: {sorted(methods)}"
        )

    merged = method_defaults(method)
    merged.update(config or {})

    # `processor` and `polling_period` configure the pipeline rather than the
    # follower, so they leave the config before a `config_passthrough` method
    # would splat whatever is left into its constructor.
    processor_type = merged.pop("processor", DEFAULT_PROCESSOR[INPUT_TYPE])
    # What the method asks to be fed with. None means it wants one MIDI
    # message at a time; a number means frames of that duration.
    wanted_polling_period = merged.pop(
        "polling_period", preferred_polling_period(method)
    )
    if polling_period is None:
        polling_period = wanted_polling_period

    context = FollowerContext(
        method=method,
        score_part=score_part,
        score_positions=score_positions,
        polling_period=polling_period,
        tempo=tempo,
        config=merged,
    )

    custom = _custom_spec(method)

    if custom is not None and custom["build_processor"] is not None:
        context.processor = custom["build_processor"](context)
    else:
        context.processor = REGISTRY.build_processor(context, processor_type)

    if custom is not None:
        if custom["build_reference"] is not None:
            context.reference_features = custom["build_reference"](context)
        else:
            context.reference_features = context.score_part.note_array()
        follower = custom["build_follower"](context)
    else:
        context.reference_features = REGISTRY.build_reference(context, method)
        follower = REGISTRY.build_follower(context, method)

    info = {
        "method": method,
        "processor": processor_type,
        # The period the follower was built with...
        "polling_period": polling_period,
        # ...and the one its spec asks to be fed at. None means the caller has
        # to hand it one MIDI message per frame.
        "wanted_polling_period": wanted_polling_period,
        "event_based": wanted_polling_period is None,
    }
    return context.processor, follower, info
