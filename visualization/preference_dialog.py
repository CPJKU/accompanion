import glob
import os

import numpy as np

from matchmaker.io.midi import get_port_names
from PyQt5 import QtWidgets


# for some reason, Mac does not like relative paths...
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(CURRENT_DIR)

MIN_WIDTH = 800
MIN_HEIGHT = 380

DEFAULT_DTW_WINDOW_SIZE = 200
DEFAULT_DTW_STEP_SIZE = 50
DEFAULT_SCORE_BPM = 60

DEFAULT_COMPOSER = "gw_full"

# DEFAULT_REFERENCES = [os.path.join(TOP_DIR, 'brahms_data', 'match', 'cc_solo',
#                                    'Brahms_Hungarian-Dance-5_Primo_2021-07-29.match'),
#                       os.path.join(TOP_DIR, 'brahms_data', 'match', 'cc_solo',
#                                    'Brahms_Hungarian-Dance-5_Primo_2021-07-27.match')]

if DEFAULT_COMPOSER == "Brahms":
    DEFAULT_REFERENCES = glob.glob(
                os.path.join(
                    TOP_DIR, 'brahms_data', 'match', "cc_solo", "*.match"
                )
            )[-15:]

    DEFAULT_PERFORMANCE = os.path.join(TOP_DIR, "brahms_data", 'midi', 'cc_solo',
                                       "Brahms_Hungarian-Dance-5_Primo_2021-07-29_w_tapping.mid")
    DEFAULT_MATCH = os.path.join(TOP_DIR, "brahms_data", 'match', 'cc_solo',
                                 "Brahms_Hungarian-Dance-5_Primo_2021-07-28.match")

    DEFAULT_MATCH_ACCOMPANIMENT = os.path.join(TOP_DIR, "brahms_data", 'trained_bm_models',
                                               "bm_predictions_2021-08-30.match")

    DEFAULT_ACCOMPANIMENT = os.path.join(TOP_DIR, "brahms_data", "musicxml",
                                         "Brahms_Hungarian-Dance-5_Secondo.musicxml")

elif DEFAULT_COMPOSER == "Schubert":
    DEFAULT_ACCOMPANIMENT = os.path.join(
        TOP_DIR, "schubert_data", "Excerpt_1", "musicxml",
        "Schubert_Rondo_E1_Secondo.musicxml"
    )
    DEFAULT_REFERENCES = glob.glob(
        os.path.join(
            TOP_DIR, "schubert_data", "Excerpt_1",  "match", "cc_primo", "*.match"
        )
    )
    DEFAULT_PERFORMANCE = os.path.join(
        TOP_DIR, "schubert_data", "Excerpt_1", "midi", "cc_solo",
        "Schubert_Rondo_E1_Primo_2021-09-28_01_w_tapping.mid"
    )
    DEFAULT_MATCH_ACCOMPANIMENT = os.path.join(
        TOP_DIR, "schubert_data", "trained_bm_models", "bm_predictions.match"
    )
    DEFAULT_MATCH = os.path.join(
        TOP_DIR, "schubert_data", "Excerpt_1", "match",
        "cc_primo", "Schubert_Rondo_E1_Primo_2021-09-28_01.match"
    )

elif DEFAULT_COMPOSER == "Schubert_long":
    DEFAULT_ACCOMPANIMENT = os.path.join(
        TOP_DIR, "schubert_data", "Gerhard_long",
            "Rondo_in_A_E2_cut62-Piano_2.musicxml"
    )
    DEFAULT_REFERENCES = glob.glob(
        os.path.join(
            TOP_DIR, "schubert_data", "Gerhard_long", "match", "*.match"
        )
    )
    DEFAULT_PERFORMANCE = os.path.join(
        TOP_DIR, "schubert_data", "Gerhard_long", "midi", "gerhard_long_4.mid"
    )
    DEFAULT_MATCH_ACCOMPANIMENT = os.path.join(
        TOP_DIR, "schubert_data", "Gerhard_long", "Schubert_Rondo_E2_cut62-2.match"
    )
    DEFAULT_MATCH = os.path.join(
        TOP_DIR, "schubert_data", "Excerpt_1", "match",
        "cc_primo", "Schubert_Rondo_E1_Primo_2021-09-28_01.match"
    )

elif DEFAULT_COMPOSER == "gw_full":
    DEFAULT_ACCOMPANIMENT = os.path.join(
        TOP_DIR, "schubert_data", "gw_full",
        "score", "Rondo_in_A_E2_cut62_final-Piano_2.musicxml")

    DEFAULT_REFERENCES = glob.glob(os.path.join(TOP_DIR, "schubert_data", "gw_full", "match", "*.match"))

    DEFAULT_PERFORMANCE = os.path.join(TOP_DIR, "schubert_data", "gw_full", "midi", "gw_final_6_no_pedal.mid")

    DEFAULT_MATCH_ACCOMPANIMENT = os.path.join(TOP_DIR, "schubert_data", "gw_full", "basismixer", "bm_v4.match")

    DEFAULT_MATCH = None
    DEFAULT_SCORE_BPM = 52

elif DEFAULT_COMPOSER == "gw_p2":
    DEFAULT_ACCOMPANIMENT = os.path.join(
        TOP_DIR, "schubert_data", "gw_p2", "score", "snippet-Piano_2.musicxml")

    DEFAULT_REFERENCES = glob.glob(os.path.join(TOP_DIR, "schubert_data", "gw_p2", "match", "*.match"))

    DEFAULT_PERFORMANCE = os.path.join(TOP_DIR, "schubert_data", "gw_p2", "midi", "gw_snippet_1.mid")

    DEFAULT_MATCH_ACCOMPANIMENT = None

    DEFAULT_MATCH = None
    DEFAULT_SCORE_BPM = 52


class PreferenceDialog(QtWidgets.QDialog):

    def __init__(self, preferences):
        super().__init__()

        self.setWindowTitle("Preferences")

        self.preferences = preferences

        # port selection
        self.port_box = QtWidgets.QComboBox()
        self.setup_available_ports()

        # file selection
        self.reference_paths = self.preferences.references
        self.acc_path = self.preferences.accompaniment
        self.perf_path = self.preferences.performance
        self.match_path = self.preferences.match
        self.accompaniment_match_path = self.preferences.accompaniment_match

        self.reference_selection = QtWidgets.QComboBox()
        self.accompaniment_selection = QtWidgets.QComboBox()
        self.performance_selection = QtWidgets.QComboBox()
        self.match_selection = QtWidgets.QComboBox()
        self.accompaniment_match_selection = QtWidgets.QComboBox()
        self.setup_file_selection()

        # score settings
        self.score_bpm = QtWidgets.QSpinBox()
        self.score_bpm.setMinimum(20)
        self.score_bpm.setMaximum(300)
        self.score_bpm.setValue(self.preferences.score_bpm)

        # dtw settings
        self.dtw_settings = QtWidgets.QWidget()
        self.dtw_tracker = QtWidgets.QComboBox()
        self.dtw_window_size = QtWidgets.QSpinBox()
        self.dtw_step_size = QtWidgets.QSpinBox()
        self.dtw_distances = QtWidgets.QComboBox()
        self.dtw_use_tempo_model = QtWidgets.QCheckBox()
        self.setup_dtw_settings()

        # setup layout
        self.setup_layout()

        # resizing
        self.setMinimumSize(MIN_WIDTH, MIN_HEIGHT)

    def setup_available_ports(self):
        port_names = get_port_names()

        if len(port_names) == 0:
            raise ValueError("No MIDI devices connected")

        for p_ix, pn in enumerate(port_names):
            self.port_box.addItem(f"{p_ix}: {pn}")

        self.port_box.setCurrentIndex(self.preferences.port_id)

    def setup_file_selection(self):

        # setup reference file selection menu
        for ref in self.reference_paths:
            self.reference_selection.addItem(os.path.basename(ref))
        self.reference_selection.addItem('Select File(s)...')
        self.reference_selection.currentTextChanged.connect(self.reference_selection_changed)

        # setup accompaniment file selection menu
        self.accompaniment_selection.addItem('Select File...')
        self.accompaniment_selection.addItem(os.path.basename(self.acc_path))
        self.accompaniment_selection.setCurrentIndex(1)
        self.accompaniment_selection.currentTextChanged.connect(self.accompaniment_selection_changed)

        # setup performance file selection menu
        self.performance_selection.addItem('Live Input')
        self.performance_selection.addItem('Select File...')
        self.performance_selection.addItem(os.path.basename(self.perf_path))
        self.performance_selection.setCurrentIndex(2)
        self.performance_selection.currentTextChanged.connect(self.performance_selection_changed)

        # setup match file selection menu
        self.match_selection.addItem('Select File...')
        if self.match_path is not None:
            self.match_selection.addItem(os.path.basename(self.match_path))
        self.match_selection.setCurrentIndex(1)
        self.match_selection.currentTextChanged.connect(self.match_selection_changed)

        # setup accompaniment match file selection menu
        self.accompaniment_match_selection.addItem('Select File...')
        if self.accompaniment_match_path is not None:
            self.accompaniment_match_selection.addItem(os.path.basename(self.accompaniment_match_path))
        self.accompaniment_match_selection.setCurrentIndex(1)
        self.accompaniment_match_selection.currentTextChanged.connect(self.accompaniment_match_selection_changed)

    def setup_dtw_settings(self):

        self.dtw_tracker.addItem("OnlineTimeWarping")
        self.dtw_tracker.addItem("TempoOnlineTimeWarping")
        self.dtw_tracker.addItem("GroundTruthTracker")

        # set default for tracker
        self.dtw_tracker.setCurrentIndex([self.dtw_tracker.itemText(i)
                                          for i in range(self.dtw_tracker.count())].index(self.preferences.dtw_tracker))

        self.dtw_window_size.setMinimum(1)
        self.dtw_window_size.setMaximum(np.iinfo(np.int32).max)
        self.dtw_window_size.setValue(self.preferences.dtw_window_size)

        self.dtw_step_size.setMinimum(1)
        self.dtw_step_size.setMaximum(np.iinfo(np.int32).max)
        self.dtw_step_size.setValue(self.preferences.dtw_step_size)

        self.dtw_distances.addItem("Manhattan")

        self.dtw_use_tempo_model.setChecked(self.preferences.dtw_tempo_model)

        QtWidgets.QWidget()
        dtw_settings_layout = QtWidgets.QGridLayout()

        dtw_settings_layout.addWidget(QtWidgets.QLabel('Tracker'), 0, 0)
        dtw_settings_layout.addWidget(self.dtw_tracker, 0, 1)

        dtw_settings_layout.addWidget(QtWidgets.QLabel('Window Size'), 1, 0)
        dtw_settings_layout.addWidget(self.dtw_window_size, 1, 1)
        dtw_settings_layout.addWidget(QtWidgets.QLabel('Step Size'), 2, 0)
        dtw_settings_layout.addWidget(self.dtw_step_size, 2, 1)
        dtw_settings_layout.addWidget(QtWidgets.QLabel('Distance Metric'), 3, 0)
        dtw_settings_layout.addWidget(self.dtw_distances, 3, 1)
        dtw_settings_layout.addWidget(QtWidgets.QLabel('Tempo Model'), 4, 0)
        dtw_settings_layout.addWidget(self.dtw_use_tempo_model, 4, 1)

        self.dtw_settings.setLayout(dtw_settings_layout)

    def setup_layout(self):
        i = 0
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.addWidget(QtWidgets.QLabel('Select Port'), i, 0)
        grid_layout.addWidget(self.port_box, i, 1)
        i += 1

        grid_layout.addWidget(QtWidgets.QLabel('Score'), i, 0)
        grid_layout.addWidget(self.reference_selection, i, 1)
        i += 1

        grid_layout.addWidget(QtWidgets.QLabel('Accompaniment'), i, 0)
        grid_layout.addWidget(self.accompaniment_selection, i, 1)
        i += 1

        grid_layout.addWidget(QtWidgets.QLabel('Performance'), i, 0)
        grid_layout.addWidget(self.performance_selection, i, 1)
        i += 1

        grid_layout.addWidget(QtWidgets.QLabel('Match (for GT)'), i, 0)
        grid_layout.addWidget(self.match_selection, i, 1)
        i += 1

        grid_layout.addWidget(QtWidgets.QLabel('Accompaniment Match'), i, 0)
        grid_layout.addWidget(self.accompaniment_match_selection, i, 1)
        i += 1

        grid_layout.addWidget(QtWidgets.QLabel('Score BPM'), i, 0)
        grid_layout.addWidget(self.score_bpm, i, 1)
        i += 1

        grid_layout.addWidget(QtWidgets.QLabel('DTW Parameters'), i, 0)
        i += 1

        grid_layout.addWidget(self.dtw_settings, i, 0)
        i += 1

        # setup ok button
        ok_button = QtWidgets.QPushButton("ok")
        ok_button.clicked.connect(lambda: self.close())
        grid_layout.addWidget(ok_button, i, 0)
        self.setLayout(grid_layout)

    def reference_selection_changed(self):

        if self.reference_selection.currentText() == "Select File(s)...":
            reference_paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Provide Reference(s)',
                                                                        os.path.join(TOP_DIR, "brahms_data"),
                                                                        "MusicXML/Match files (*.musicxml *.match)",
                                                                        options=QtWidgets.QFileDialog.DontUseNativeDialog)

            self.reference_paths = reference_paths

            self.reference_selection.clear()
            for ref in self.reference_paths:
                self.reference_selection.addItem(os.path.basename(ref))
            self.reference_selection.addItem('Select File(s)...')

            self.reference_selection.setCurrentIndex(0)

    def accompaniment_selection_changed(self):

        if self.accompaniment_selection.currentText() == "Select File...":
            self.acc_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Provide Accompaniment',
                                                                     os.path.join(TOP_DIR, "brahms_data"),
                                                                     "MusicXML files (*.musicxml)",
                                                                     options=QtWidgets.QFileDialog.DontUseNativeDialog)

            self.accompaniment_selection.removeItem(1)
            self.accompaniment_selection.addItem(os.path.basename(self.acc_path))
            self.accompaniment_selection.setCurrentIndex(1)

    def performance_selection_changed(self):

        txt = self.performance_selection.currentText()
        if txt == "Live Input":
            # live input
            self.perf_path = None
        elif txt == "Select File...":
            self.performance_selection.removeItem(2)
            self.perf_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Provide Performance',
                                                                      os.path.join(TOP_DIR, "brahms_data"),
                                                                      "MIDI files (*.mid)",
                                                                      options=QtWidgets.QFileDialog.DontUseNativeDialog)
            self.performance_selection.addItem(self.perf_path)
            self.performance_selection.setCurrentIndex(2)

    def match_selection_changed(self):

        if self.match_selection.currentText() == "Select File...":
            self.match_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Provide Matchfile',
                                                                       os.path.join(TOP_DIR, "brahms_data"),
                                                                       "Match files (*.match)",
                                                                       options=QtWidgets.QFileDialog.DontUseNativeDialog)

            self.match_selection.removeItem(1)
            self.match_selection.addItem(os.path.basename(self.match_path))
            self.match_selection.setCurrentIndex(1)

    def accompaniment_match_selection_changed(self):

        if self.accompaniment_match_selection.currentText() == "Select File...":
            self.accompaniment_match_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Provide Matchfile',
                                                                       os.path.join(TOP_DIR, "brahms_data", "trained_bm_models"),
                                                                       "Match files (*.match)",
                                                                       options=QtWidgets.QFileDialog.DontUseNativeDialog)

            self.accompaniment_match_selection.removeItem(1)
            self.accompaniment_match_selection.addItem(os.path.basename(self.accompaniment_match_path))
            self.accompaniment_match_selection.setCurrentIndex(1)

    def get_port_id(self):
        return self.port_box.currentIndex()

    def get_score_bpm(self):
        return self.score_bpm.value()

    def get_reference_paths(self):
        return self.reference_paths

    def get_accompaniment_path(self):
        return self.acc_path

    def get_performance_path(self):
        return self.perf_path

    def get_match_path(self):
        return self.match_path

    def get_accompaniment_match_path(self):
        return self.accompaniment_match_path

    def get_dtw_params(self):
        params = dict(window_size=self.dtw_window_size.value(),
                      step_size=self.dtw_step_size.value(),
                      local_cost_fun=self.dtw_distances.currentText())

        return params

    def get_tracker_type(self):
        return self.dtw_tracker.currentText()

    def use_tempo_model(self):
        return self.dtw_use_tempo_model.isChecked()
