import partitura
from misc.partitura_utils import partitura_to_framed_midi_custom

if __name__ == "__main__":

    # See main.py to see how to load performances,
    # set up the score follower and all initializations
    # In this case we don't need to have a sequencer
    # load scores and performances (see `setup_scores` method in main.py)

    # This is the midi file containing the performance of the solo part
    midi_fn = ""

    # the "object" holding the accompaniment score, which is an instance of
    # `AccompanimentScore`
    acc_score = None
    # Also, load the ground truth performance of the accompaniment
    accompaniment_match = ""
    accompaniment_ppart, accompaniment_alignment = partitura.load_match(
        accompaniment_match_fn, first_note_at_zero=True
    )
    # create score follower
    pipeline = None
    score_follower = None

    polling_period = 0.01

    # generate frames
    frames = partitura_to_framed_midi_custom(
        midi_fn, pipeline=pipeline, polling_period=polling_period, is_performance=True
    )

    # This code was copied and sliglty adapted from main.py, a few
    # things might need to be set up. for it to work

    perf_start = False
    for ix, output in enumerate(frames):
        # change to corresponding time of the frames
        solo_p_onset = ix * polling_period

        if not perf_start and (output > 0).any():
            perf_start = True

        if perf_start:
            score_idx, score_position = score_follower(output)

            solo_s_onset, onset_index = onset_tracker(score_position)

            if solo_s_onset is not None:
                if onset_index == 0:
                    expected_solo_onset = 0
                else:

                    # this are the tempo models that we want
                    # to test
                    beat_period, expected_solo_onset = tempo_model(
                        solo_p_onset, solo_s_onset
                    )

                accompanist.accompaniment_step(
                    solo_s_onset=solo_s_onset,
                    solo_p_onset=expected_solo_onset,
                    velocity=60,
                    beat_period=beat_period,
                    articulation=1.0,
                )

    """
    TODO:
    0. adjust this script to be used for all midi performances of the solo part (I will split the midi performances (of the full piece) and create the matchfiles later today or tomorrow)
    1. generate a note array from acc_score (like the note arrays from partitura?), to compare the generated accompaniment to the accompaniment from the matchfiles
    2. Using `accompaniment_alignment` and `accompaniment_ppart` get the score notes in the accompaniment that were actually played (e.g., ignore insertions and do not use skipped notes`
    3. compare the onset times of the notes in `accompaniment_ppart` and the corresponding notes from the note array generated from `acc_score`.
    4. Get statistics (mean, median, std)
    5. Do the script for different combinations of tempo_models (see models in `accompanion.tempo_models` and their respective parameters).
    """
