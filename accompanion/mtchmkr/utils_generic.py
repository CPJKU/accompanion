# -*- coding: utf-8 -*-
"""
Generic utilities

TODO
----
* Use a madmom like format
"""
import partitura


# Use madmom's SequentialProcessor?
class SequentialOutputProcessor(object):
    """"""

    def __init__(self, processors):
        self.processors = list(processors)

    def __call__(self, data, **kwargs):
        for processor in self.processors:
            data, kwargs = processor(data, kwargs)
        return data

    def reset(self):
        for processor in self.processors:
            if hasattr(processor, "reset"):
                processor.reset()


def matchfile_to_midi(fn, perf_outfile, score_outfile=None):
    """
    create a MIDI file from a Matchfile

    Parameters
    ----------
    fn : filename
        Match file to be converted to MIDI.
    perf_outfile : filename
        MIDI file which will contain the perforamance information from
        the input Match file.
    score_outfile : filename or None
        MIDI file wich will contain the score information from the input
        Match file. This file will only be generated if `score_outfile` is
        not None.
    """
    create_spart = score_outfile is not None

    match_info = partitura.load_match(fn=fn, create_part=create_spart)

    ppart, _ = match_info[0], match_info[1]
    partitura.save_performance_midi(ppart, perf_outfile)

    if create_spart:
        spart = match_info[2]
        partitura.save_score_midi(spart, score_outfile)
