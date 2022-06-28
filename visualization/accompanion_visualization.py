
import multiprocessing
import platform
import sys
import warnings

import numpy as np

from accompanion.main import ACCompanion, ACC_PROCESS

from collections import deque
from multiprocessing import Pipe
from queue import Queue
from PyQt5 import QtCore, QtWidgets
from pyqtgraph import PlotWidget, plot

from visualization.midi_helper import MIDI_KEYS
from visualization.piano_roll_visualization import (PianoRollVisualization, SCORE_COLOR,
                                                    ACCOMPANIMENT_COLOR, PERFORMANCE_COLOR)
from visualization.preference_dialog import (PreferenceDialog, DEFAULT_REFERENCES, DEFAULT_ACCOMPANIMENT,
                                             DEFAULT_PERFORMANCE, DEFAULT_MATCH, DEFAULT_MATCH_ACCOMPANIMENT,
                                             DEFAULT_SCORE_BPM, DEFAULT_DTW_STEP_SIZE, DEFAULT_DTW_WINDOW_SIZE)

FRAME_RESOLUTION = 0.01  # 100 frames per second
WINDOW_HEIGHT = 1050
WINDOW_WIDTH = 1280
MAX_SCORE_PR_LENGTH = 2000
MAX_PERF_PR_LENGTH = 1000

PLATFORM = platform.system()

if PLATFORM == "Linux":
    MIDI_DRIVER = "alsa"
elif PLATFORM == "Darwin":
    MIDI_DRIVER = "coreaudio"
elif PLATFORM == "Windows":
    # raise ValueError("Windows is not supported yet...")
    MIDI_DRIVER = None

if PLATFORM != "Windows":
    from accompanion.fluid import FluidsynthPlayer
else:
    class FluidsynthPlayer(object):
        def __init__(self):
            self.name = "FluidsynthPlayer dummy"
        def send(self, message):
            warnings.warn(f"Fluidsynth Player is currently not supported under windows, use a MIDI output port!")


class PipeQueue(Queue):
    def __init__(self):
        Queue.__init__(self)

    def recv(self):
        return self.get()

    def poll(self):
        return self.not_empty

    def send(self, item, block=True, timeout=None):
        super(PipeQueue, self).put(item, block, timeout)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, preferences, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.timer = None

        self.accompanion = None

        self.headless = preferences.headless

        # piano rolls for visualization
        self.score_piano_roll, self.performance_piano_roll,  self.accompaniment_piano_roll = None, None, None

        self.tempi = None
        self.tempo_widget = PlotWidget()
        self.tempo_plot = self.tempo_widget.plot()
        self.tempo_widget.setTitle('Tempo Curve')
        self.tempo_widget.setLabel('left', 'Tempo')

        # input data dialog
        self.preference_dialog = PreferenceDialog(preferences)

        # score/performance canvas
        self.score_frame = PianoRollVisualization()
        self.perf_frame = PianoRollVisualization()

        # start/stop buttons
        self.start_tracking_button = QtWidgets.QPushButton("Start tracking...")
        self.start_tracking_button.clicked.connect(self.start_tracking)

        self.stop_tracking_button = QtWidgets.QPushButton("Stop tracking...")
        self.stop_tracking_button.clicked.connect(self.stop_tracking)

        self.setup_layout()
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        if not self.headless:
            self.show()

    def setup_layout(self):
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.score_frame)
        layout.addWidget(self.perf_frame)
        layout.addWidget(self.tempo_widget)
        layout.addWidget(self.start_tracking_button)
        layout.addWidget(self.stop_tracking_button)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def stop_tracking(self):
        if self.timer is not None:
            self.timer.stop()

        if self.accompanion is not None:
            self.accompanion.terminate()
            self.accompanion = None


    def start_tracking(self):

        self.stop_tracking()

        if not self.headless:
            self.preference_dialog.exec()

        # prepare empty performance and accompaniment frames
        self.performance_piano_roll = deque(maxlen=MAX_PERF_PR_LENGTH)
        self.accompaniment_piano_roll = deque(maxlen=MAX_PERF_PR_LENGTH)
        self.tempi = deque([120] * MAX_PERF_PR_LENGTH, maxlen=MAX_PERF_PR_LENGTH)

        for i in range(self.performance_piano_roll.maxlen):
            # self.performance_piano_roll.append(np.ones(MIDI_KEYS, dtype=np.uint8) * 255)
            self.performance_piano_roll.append(np.zeros(MIDI_KEYS, dtype=np.uint8))
            # self.accompaniment_piano_roll.append(np.ones(MIDI_KEYS, dtype=np.uint8) * 255)
            self.accompaniment_piano_roll.append(np.zeros(MIDI_KEYS, dtype=np.uint8))

        if self.preference_dialog.get_performance_path() is not None:
            acc_output_to_sound_port_name = FluidsynthPlayer
        else:
            acc_output_to_sound_port_name = self.preference_dialog.get_port_id()

        router_kwargs = dict(
            solo_input_to_accompaniment_port_name=self.preference_dialog.get_port_id(),
            acc_output_to_sound_port_name=acc_output_to_sound_port_name,
            MIDIPlayer_to_sound_port_name=FluidsynthPlayer,
            MIDIPlayer_to_accompaniment_port_name=self.preference_dialog.get_port_id(),
            simple_button_input_port_name=None,
        )
        # router_kwargs = dict(
        #     solo_input_to_accompaniment_port_name="M-Audio MIDISPORT Uno",# "Clavinova", # "Silent Piano",
        #     acc_output_to_sound_port_name="M-Audio MIDISPORT Uno", # "Silent Piano", # fluidsynth
        #     # MIDIPlayer_to_sound_port_name="hshs",
        #     # MIDIPlayer_to_accompaniment_port_name="shsh",
        #     # simple_button_input_port_name=None,
        # )


        #router_kwargs = dict(
        #3    solo_input_to_accompaniment_port_name="Scarlett",# "USB-MIDI", # "Clavinova", # "Silent Piano",
        #    #acc_output_to_sound_port_name="M-Audio MIDISPORT Uno",
        #    acc_output_to_sound_port_name="Clavinova",
        #    MIDIPlayer_to_sound_port_name=None,
        #    MIDIPlayer_to_accompaniment_port_name=None,
        #    simple_button_input_port_name=None,
        #)

        if ACC_PROCESS:
            self.p_output, self.p_input = Pipe()
        else:
            queue = PipeQueue()
            self.p_output = queue
            self.p_input = queue

        tempo_tapping = (3, "eighth")
        self.accompanion = ACCompanion(
            solo_fn=self.preference_dialog.get_reference_paths(),
            acc_fn=self.preference_dialog.get_accompaniment_path(),
            midi_fn=self.preference_dialog.get_performance_path(),
            router_kwargs=router_kwargs,
            init_bpm=self.preference_dialog.get_score_bpm(),
            polling_period=FRAME_RESOLUTION,
            follower=self.preference_dialog.get_tracker_type(),
            follower_kwargs=self.preference_dialog.get_dtw_params(),
            ground_truth_match=self.preference_dialog.get_match_path(),
            accompaniment_match=self.preference_dialog.get_accompaniment_match_path(),
            pipe=self.p_output,
            use_ceus_mediator=False,
            tempo_tapping=tempo_tapping,
            adjust_following_rate=0.2
        )

        self.accompanion.start()
        self.score_piano_roll = (255 - self.p_input.recv().T * 255).astype(np.uint8)[::-1]
        self.plot_canvas()

        # Setup a timer to trigger the tracking.
        if not self.headless:
            self.timer = QtCore.QTimer()
            self.timer.setInterval(1)
            self.timer.timeout.connect(self.update_visualization)
            self.timer.start()

    def plot_canvas(self, score_idx=None):

        # plot performance
        self.perf_frame.update_pr(np.stack((self.performance_piano_roll, self.accompaniment_piano_roll)).transpose((0, 2, 1)),
                                  midi_color=[PERFORMANCE_COLOR, ACCOMPANIMENT_COLOR])

        # plot score
        if score_idx is None:
            score_idx = 0

        # compute score window
        from_ = max(0, score_idx - MAX_SCORE_PR_LENGTH // 2)
        to_ = min(score_idx + MAX_SCORE_PR_LENGTH // 2, self.score_piano_roll.shape[1])
        score = self.score_piano_roll[:, from_:to_]

        # ensure the size of the score window
        pad = MAX_SCORE_PR_LENGTH - score.shape[1]

        if pad > 0:
            if from_ == 0:
                # pad beginning of score
                score = np.pad(score, ((0, 0), (pad, 0)), 'constant', constant_values=255)
            else:
                # pad end of score
                score = np.pad(score, ((0, 0), (0, pad)), 'constant', constant_values=255)

                # set padding to 0 as no index shifting is necessary
                pad = 0

        # adapt score and ground truth index based on window
        score_idx = score_idx + pad - from_

        self.score_frame.update_pr(score, pred_idx=score_idx, midi_color=SCORE_COLOR)
        self.tempo_plot.setData(y=self.tempi)

    def update_visualization(self):

        if self.p_input.poll():
            frame, accompaniment_frame, index, tempo = self.p_input.recv()
            self.tempi.append(tempo)
            # prepare and add frame to performance piano roll
            self.performance_piano_roll.append(frame[::-1].astype(np.uint8))

            # prepare and add frame to  accompaniment piano roll
            self.accompaniment_piano_roll.append(accompaniment_frame[::-1].astype(np.uint8))

            # plot score and performance
            self.plot_canvas(index)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Accompanion')
    parser.add_argument('--port_id', help='MIDI port id.', type=int, default=0)
    parser.add_argument('--references', help='path to reference files', type=str, default=DEFAULT_REFERENCES, nargs='+')
    parser.add_argument('--accompaniment', help='path to accompaniment file', type=str, default=DEFAULT_ACCOMPANIMENT)
    parser.add_argument('--performance', help='path to performance file ("live" for live input)',
                        type=str, default=DEFAULT_PERFORMANCE)
    parser.add_argument('--match', help='path to match file (for GT tracking)',
                        type=str, default=DEFAULT_MATCH)
    parser.add_argument('--accompaniment_match', help='path to match file (for accompaniment)',
                        type=str, default=DEFAULT_MATCH_ACCOMPANIMENT)
    parser.add_argument('--score_bpm', help='default score bpm', type=int, default=DEFAULT_SCORE_BPM)
    parser.add_argument('--dtw_window_size', help='DTW window size', type=int, default=DEFAULT_DTW_WINDOW_SIZE)
    parser.add_argument('--dtw_step_size', help='DTW step size', type=int, default=DEFAULT_DTW_STEP_SIZE)
    parser.add_argument('--dtw_tempo_model', help='DTW tempo model', default=False, action='store_true')
    parser.add_argument('--dtw_tracker', choices=['OnlineTimeWarping', 'TempoOnlineTimeWarping', 'GroundTruthTracker'],
                        default='OnlineTimeWarping')

    parser.add_argument('--headless', help='execute without GUI', default=False, action='store_true')
    args = parser.parse_args()

    if PLATFORM == "Darwin" or PLATFORM == "Linux":
        multiprocessing.set_start_method('spawn')

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(args)

    if args.headless:
        w.start_tracking()
    else:
        app.exec_()
