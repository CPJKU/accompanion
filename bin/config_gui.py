# -*- coding: utf-8 -*-
"""
Configuration GUI for the ACCompanion.

The GUI asks two things, in two windows:

1.  *Which ACCompanion, and which score follower?* The ACCompanion ships two
    followers of its own -- the HMM and on-line time warping -- and, through
    `accompanion.matchmaker_accompanion.MatchmakerACCompanion`, every score
    follower the Matchmaker package provides. The list of Matchmaker methods is
    read live from Matchmaker's registry, so a method added there (in
    ``methods.yaml`` or through ``matchmaker.register_method``) shows up here
    without any change to this file.

2.  *How should it be configured?* The second window is generated from the
    chosen class' constructor signature, grouped into tabs. Fields that have a
    small, known set of valid values -- MIDI ports, tempo models, score
    followers -- are rendered as drop-downs rather than as free text, through
    the `Hook` system described below.

The generation itself works in three steps:

    *   the constructor's parameters and their defaults are turned into a tree
        of `ConfigurationNode`s, one node per parameter, with a child node per
        entry of a parameter that is a dict;

    *   the tree is turned into a PySimpleGUI layout, one row per leaf;

    *   once the user is done, the strings collected from the widgets are
        evaluated back into Python objects and gathered into the dict that is
        splatted into the constructor.

`Hook` objects override any of those three steps for a parameter, which is how
the port, tempo model and score follower drop-downs are built.
"""
import os
from ast import literal_eval
from inspect import Parameter, signature
from typing import Union

import PySimpleGUI as sg
import yaml

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------

#: Amber on slate, the colours the fullscreen ACCompanion app uses.
ACCENT = "#ffbd09"
BACKGROUND = "#1f2732"
FIELD = "#2f3a49"
BORDER = "#3c4959"
TEXT = "#e7ecf2"
MUTED = "#95a3b5"
DANGER = "#ff7b72"
SUCCESS = "#7ddc8f"

THEME_NAME = "ACCompanion"

FONT_FAMILY = "Helvetica"
FONT = (FONT_FAMILY, 10)
FONT_BOLD = (FONT_FAMILY, 10, "bold")
FONT_SMALL = (FONT_FAMILY, 9)
FONT_TITLE = (FONT_FAMILY, 20, "bold")
FONT_MONO = ("Courier", 9)

#: Width of the label column, in characters, so that every field lines up.
LABEL_WIDTH = 26
#: Width of the input column, in characters.
FIELD_WIDTH = 38
#: Width of the muted caption under a field, in characters.
CAPTION_WIDTH = 62

_TAB_SIZE = (900, 430)

_theme_applied = False


def apply_theme():
    """Register and select the ACCompanion's PySimpleGUI theme."""
    global _theme_applied

    if _theme_applied:
        return

    if THEME_NAME not in sg.theme_list():
        sg.theme_add_new(
            THEME_NAME,
            {
                "BACKGROUND": BACKGROUND,
                "TEXT": TEXT,
                "INPUT": FIELD,
                "TEXT_INPUT": TEXT,
                "SCROLL": FIELD,
                "BUTTON": (BACKGROUND, ACCENT),
                "PROGRESS": (ACCENT, FIELD),
                "BORDER": 0,
                "SLIDER_DEPTH": 0,
                "PROGRESS_DEPTH": 0,
            },
        )

    sg.theme(THEME_NAME)
    # Tk's default ttk theme ignores the colours set on a combo box, leaving
    # every drop-down a light grey box on the dark background; clam honours
    # them.
    sg.set_options(font=FONT, tooltip_font=FONT_SMALL, ttk_theme=sg.THEME_CLAM)

    # Scrollbars take the button colours by default, which makes every one of
    # them a bright amber bar. They are furniture, not calls to action.
    sg.ttk_part_mapping_dict.update(
        {
            sg.TTK_SCROLLBAR_PART_TROUGH_COLOR: BACKGROUND,
            sg.TTK_SCROLLBAR_PART_BACKGROUND_COLOR: FIELD,
            sg.TTK_SCROLLBAR_PART_ARROW_BUTTON_ARROW_COLOR: MUTED,
            sg.TTK_SCROLLBAR_PART_FRAME_COLOR: BACKGROUND,
            sg.TTK_SCROLLBAR_PART_RELIEF: sg.RELIEF_FLAT,
            sg.TTK_SCROLLBAR_PART_SCROLL_WIDTH: 10,
        }
    )

    _theme_applied = True


def _restyle_combos(window):
    """Give the drop-downs the field colour when they hold the focus.

    PySimpleGUI paints a read-only combo's selection highlight in the text
    colour, which on a dark theme lights up the whole field.
    """
    for element in window.element_list():
        if not isinstance(element, sg.Combo):
            continue

        style = getattr(element, "ttk_style", None)

        if style is None:
            continue

        name = element.Widget.cget("style")

        style.configure(name, selectbackground=FIELD, selectforeground=TEXT)
        style.map(
            name,
            foreground=[("readonly", TEXT)],
            fieldbackground=[("readonly", FIELD)],
            selectbackground=[("readonly", FIELD)],
            selectforeground=[("readonly", TEXT)],
        )

        # A read-only combo keeps its text selected, which Tk paints in its own
        # colours whatever the style says. Dropping the selection leaves the
        # field looking like the field it is.
        widget = element.Widget
        widget.selection_clear()
        widget.bind(
            "<FocusIn>",
            lambda event: event.widget.after_idle(event.widget.selection_clear),
            add="+",
        )
        widget.bind(
            "<<ComboboxSelected>>",
            lambda event: event.widget.selection_clear(),
            add="+",
        )


def primary_button(text, key, **kwargs):
    """A call-to-action button: dark text on amber."""
    return sg.Button(
        text,
        key=key,
        font=FONT_BOLD,
        button_color=(BACKGROUND, ACCENT),
        border_width=0,
        pad=((6, 0), (0, 0)),
        **kwargs,
    )


def secondary_button(text, key, **kwargs):
    """A quiet button: light text on the field colour."""
    return sg.Button(
        text,
        key=key,
        font=FONT,
        button_color=(TEXT, FIELD),
        border_width=0,
        pad=((6, 0), (0, 0)),
        **kwargs,
    )


def _combo(values, default_value, key, width=FIELD_WIDTH, **kwargs):
    """A read-only drop-down, themed to match the input fields."""
    return sg.Combo(
        list(values),
        default_value=default_value,
        key=key,
        size=(width, 1),
        readonly=True,
        background_color=FIELD,
        text_color=TEXT,
        button_background_color=FIELD,
        button_arrow_color=ACCENT,
        **kwargs,
    )


def _update_combo(element, **kwargs):
    """Change a drop-down's contents without leaving its text selected."""
    element.update(**kwargs)
    element.Widget.selection_clear()


def _caption(text, width=CAPTION_WIDTH, key=None):
    """A muted explanatory line, indented under the field it describes."""
    return [
        sg.Text("", size=(LABEL_WIDTH, 1)),
        sg.Text(
            text,
            size=(width, max(1, len(text) // width + 1)),
            font=FONT_SMALL,
            text_color=MUTED,
            key=key,
        ),
    ]


def _section(title, rows, nested=False):
    """A titled frame holding the rows of a composite parameter.

    Only the outermost group is outlined; a group inside one is set apart by
    its heading alone, so that the frames do not stack up into a box in a box.
    """
    return sg.Frame(
        title,
        rows,
        font=FONT_BOLD,
        title_color=ACCENT,
        relief=sg.RELIEF_FLAT if nested else sg.RELIEF_SOLID,
        border_width=0 if nested else 1,
        expand_x=True,
        pad=((0, 0), (4, 4 if nested else 12)),
    )


# ---------------------------------------------------------------------------
# Reading a constructor
# ---------------------------------------------------------------------------


def load_class(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])

    return getattr(module, class_name)


def class_init_args(class_init):
    """The parameters of an ``__init__``, without ``self``."""
    s = signature(class_init)

    return [p for p in s.parameters.values()][1:]


def constructor_parameters(class_object):
    """Every keyword the class' constructor accepts, ``**kwargs`` included.

    `MatchmakerACCompanion` declares four parameters of its own and hands
    everything else to `HMMACCompanion`, so reading its signature alone would
    offer four fields. Wherever a constructor collects ``**kwargs``, the
    parameters of the constructor it inherits from are added to the list.
    """
    collected = {}

    for cls in class_object.__mro__:
        if cls is object:
            break

        class_init = cls.__dict__.get("__init__")

        if class_init is None:
            continue

        parameters = class_init_args(class_init)
        forwards = any(p.kind is Parameter.VAR_KEYWORD for p in parameters)

        for p in parameters:
            if p.kind in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL):
                continue
            collected.setdefault(p.name, p)

        if not forwards:
            break

    return list(collected.values())


def default_instance(t):
    """An empty value of the annotated type, used when a parameter has none."""

    def get_origin(t):
        return getattr(t, "__origin__", None)

    def get_args(t):
        return getattr(t, "__args__", ())

    origin = get_origin(t)

    if origin is Union:
        # Optional[X]: the empty value of whatever X is.
        annotated = [a for a in get_args(t) if a is not type(None)]

        return default_instance(annotated[0]) if len(annotated) > 0 else ""
    elif origin is not None:
        # A parameterised generic such as Dict[str, Any]: the alias itself
        # cannot be instantiated, its origin can.
        return origin()
    elif t in (str, os.PathLike):
        return ""
    else:
        return t()


def constructor_defaults(class_object):
    """``{parameter name: default value}`` for a constructor.

    A parameter that defaults to None, or has no default at all, is given the
    empty value of its annotation, so that the GUI has something to show.
    """
    from copy import deepcopy

    defaults = {}

    for p in constructor_parameters(class_object):
        if p.default not in (p.empty, None):
            # A copy: the ACCompanion pops entries out of the dicts it is
            # handed, and these are the class' own default objects.
            defaults[p.name] = deepcopy(p.default)
        elif p.annotation is not p.empty:
            defaults[p.name] = default_instance(p.annotation)
        else:
            defaults[p.name] = ""

    return defaults


def currently_supported_types(t):
    return t in (int, float, bool, list, dict, str)


# ---------------------------------------------------------------------------
# The configuration tree
# ---------------------------------------------------------------------------


class ConfigurationNode(object):
    __slots__ = ["type", "child_names_and_children", "data"]

    """
    This class is for structuring the configuration process as
        *	building a tree whose Nodes contain
            -	the type of the parameter that can be configured
            - 	the data that will be used for the end configuration (for example, an int for a 'size' parameter or a list of floats for a 'samples' parameter)
            -	child-Nodes if the parameter is a composite object of configurable parameters (currently, only dicts are supported)

        *	transforming the tree into a PySimpleGUI layout via recursively transforming subtrees into sublayouts and integrating them into the overall layout

        *	setting the 'data' attribute of a parameter via searching for it in the tree and evaluating the associated string gathered from PySimpleGUI

        *	once the configuration is accepted by the user, the tree is evaluated by recursively transforming subtrees into primitive types and dicts
            and then gathering those values along with their names in a dict
            (only dicts are currently supported and the reason is that Python objects ultimately are dicts with syntactic sugar and dicts can be easily passed
            to a function or init-method via **)
            via the type_checked flag, users can set if before evaluation, the tree should be recursively checked if the type of 'data' aligns with 'type'

    Attributes:
        type:						Python type object (like, int, dict, type, etc.)

        child_names_and_children:	list[(child_name: str, child: ConfigurationNode)]
            currently, names are not part of Nodes themselves, but are stored paired with the associated child since this way Nodes can have multiple names
            however, this might change in the future

        data:						Python object
            currently, data and child_names_and_children are supposed to exclude each other from being set to a non-None value
            meaning, if data is not None, then child_names_and_children is None, and vice versa
    """

    def __init__(self, node_type, child_names_and_children=None, data=None):
        self.type = node_type
        self.child_names_and_children = (
            [] if child_names_and_children is None else child_names_and_children
        )
        self.data = data

    def value(self):
        if self.type is dict and len(self.child_names_and_children) > 0:
            return {
                child_name: child.value()
                for child_name, child in self.child_names_and_children
            }
        else:
            return self.data

    def search(self, search_name):
        dot_loc = search_name.find(".")

        outer_scope = search_name[:dot_loc] if dot_loc >= 0 else search_name

        for child_name, child in self.child_names_and_children:
            if outer_scope == child_name:
                if dot_loc < 0:
                    return child
                if child.type is dict:
                    return child.search(search_name[dot_loc + 1 :])
                else:
                    return None

        return None


def check_for_type_error(config_node, enclosing_scope=""):
    if not config_node.type is dict:
        if len(config_node.child_names_and_children) > 0:
            raise TypeError(
                f"Node error at {enclosing_scope[1:]}: node is not of type dict, "
                f"but has children {config_node.child_names_and_children}"
            )
        elif type(config_node.data) != config_node.type:
            raise TypeError(
                f"{enclosing_scope[1:]} should be of type "
                f"{config_node.type.__name__}, but {config_node.data!r} is of "
                f"type {type(config_node.data).__name__}"
            )
    elif len(config_node.child_names_and_children) > 0:
        if not config_node.data is None:
            raise TypeError(
                f"Node error at {enclosing_scope[1:]}: node has children "
                f"{config_node.child_names_and_children}, but also data "
                f"{config_node.data}"
            )

        for child_name, child in config_node.child_names_and_children:
            check_for_type_error(child, enclosing_scope + "." + child_name)
    elif not type(config_node.data) is dict:
        raise TypeError(
            f"{enclosing_scope[1:]} should be a dict, but {config_node.data!r} "
            f"is of type {type(config_node.data).__name__}"
        )


class Hook(object):
    __slots__ = ("trigger", "configuration", "layout", "update", "evaluation")
    """
    A Hook object is intended to make it possible for users to provide or override functionality in the configuration GUI system

    Attributes:
        configuration: object -> ConfigurationNode
            this function is intended for transforming python objects into ConfigurationNodes which aren't currently supported (see currently_supported_types)
            can also be used to override the default transformation of supported types


        layout: (ConfigurationNode, enclosing_scope: str) -> list[list[PySimpleGUI.Element]]
            PySimpleGUI works by defining a grid of Elements
            this function transforms a ConfigurationNode into a list of rows of PySimpleGUI.Elements,
                which then get inserted into the overall layout of the GUI wherever the trigger occured
            with this function the look and interaction of/with parameters can be customized
                (for example,
                    use a FileBrowse-Element for choosing a file,
                    use a Combo-Element if valid values are of a small, finite size and can be chosen from a list,
                    etc.
                )

        update: (PySimpleGUI.Window, event: str, values: dict) -> str or None
            this function is somewhat of a workaround of PySimpleGUIs limited capabilities
            window.read is used in the configuration GUI and that method returns an event (in the form of a string)
                and	either a list or a dict of values (usually a dict, for details see PySimpleGUI docs)
            using the event and values dict, one can update the GUI window accordingly (for details on how to update a PySimpleGUI.Window, see the docs)
            this function is intended to adapt the window layout during configuration while the layout function is for defining the initial layout
            this separation makes sense since PySimpleGUI offers limited ways to change a window once it has been initialized
            whatever string it returns is shown in the status bar at the bottom of the window

        evaluation: (config_string: str) -> object
            the configurations done by the user are done over strings
            this function is intended to transform the result of such a configuration string into a proper Python object which isn't currently supported (see currently_supported_types)
            can also be used to override the default transformation of supported types (currently this is ast.literal_eval for non-string types)

        trigger : (name:str, data_type:type, data: object) -> bool
            boolean function which determines if the other functions should be employed in a certain context which is represented by the input
            name is similiar to accessing nested objects, i.e. 'person.birth_date.year'
            for example, the trigger for general parameters with the name 'scale' and the type float would look like this
                lambda n,t,o: n.split('.')[-1]=='scale' and t is float
            and the trigger for the specific parameter 'cube.scale' would look like this
                lambda n,t,o: n=='cube.scale'
    """

    def __init__(
        self, trigger, configuration=None, layout=None, update=None, evaluation=None
    ):
        if trigger is None:
            raise ValueError("a Hook object needs a trigger")

        if (
            configuration is None
            and layout is None
            and update is None
            and evaluation is None
        ):
            raise ValueError(
                "a Hook object needs either a configuration, layout, update or evaluation function"
            )

        self.trigger = trigger
        self.configuration = configuration
        self.layout = layout
        self.update = update
        self.evaluation = evaluation


def _no_hooks():
    return dict(triggers=[], functions=[])


def _retrieve(full_name, data_type, data, hooks):
    if hooks is None:
        return None

    for i, trigger in enumerate(hooks["triggers"]):
        if trigger(full_name, data_type, data):
            return hooks["functions"][i]
    return None


def configuration_tree(underlying_dict, configuration_hooks=None, enclosing_scope=""):
    child_names_and_children = []

    for k, v in underlying_dict.items():

        configure = _retrieve(enclosing_scope + k, type(v), v, configuration_hooks)

        if configure is None and not currently_supported_types(type(v)):
            print(
                f"the configuration GUI currently doesn't support parameters of type {type(v)} and therefore silently ignores {enclosing_scope + k}"
            )
            continue

        if not configure is None:
            child = configure(v)
        elif type(v) is dict and len(v) > 0:
            child = configuration_tree(
                v, configuration_hooks, enclosing_scope + k + "."
            )
        else:
            child = ConfigurationNode(type(v), data=v)

        child_names_and_children.append((k, child))

    return ConfigurationNode(dict, child_names_and_children=child_names_and_children)


# ---------------------------------------------------------------------------
# What the fields are called, and what they mean
# ---------------------------------------------------------------------------

#: Names that do not read well when their underscores are simply removed.
_DISPLAY_NAMES = {
    "solo_fn": "Solo score",
    "acc_fn": "Accompaniment score",
    "accompaniment_match": "Accompaniment match",
    "midi_fn": "MIDI file instead of a soloist",
    "init_bpm": "Initial tempo (BPM)",
    "init_velocity": "Initial velocity",
    "use_ceus_mediator": "Use CEUS mediator",
    "bypass_audio": "Bypass audio",
    "test": "Dummy MIDI routing",
    "record_midi": "Record MIDI",
    "score_follower_kwargs": "Score follower",
    "midi_router_kwargs": "MIDI ports",
    "tempo_model_kwargs": "Tempo model",
    "performance_codec_kwargs": "Performance codec",
    "accompanist_decoder_kwargs": "Accompanist decoder",
    "input_processor": "Input processor",
    "processor_kwargs": "Processor options",
    "score_follower_kwargs.score_follower": "Method",
    "score_follower_kwargs.score_follower_kwargs": "Method options",
    "tempo_model_kwargs.tempo_model": "Model",
    "MIDIPlayer_to_sound_port_name": "MIDI player to sound",
    "MIDIPlayer_to_accompaniment_port_name": "MIDI player to follower",
    "solo_input_to_accompaniment_port_name": "Solo input",
    "acc_output_to_sound_port_name": "Accompaniment output",
    "simple_button_input_port_name": "Button input",
    "velocity_trend_ma_alpha": "Velocity trend smoothing",
    "articulation_ma_alpha": "Articulation smoothing",
    "velocity_dev_scale": "Velocity deviation scale",
    "velocity_min": "Minimum velocity",
    "velocity_max": "Maximum velocity",
    "velocity_solo_scale": "Solo velocity carry-over",
    "log_articulation_scale": "Articulation scale",
    "mechanical_delay": "Mechanical delay (s)",
    "polling_period": "Polling period (s)",
}

#: One line per field, shown under it and as its tooltip. Keyed by the full
#: dotted path, or by the bare parameter name where the meaning is the same
#: wherever it appears.
_HELP = {
    "solo_fn": "MusicXML score of the part the soloist plays.",
    "acc_fn": "MusicXML score of the part the ACCompanion plays.",
    "accompaniment_match": "Optional match file of recorded performances, used to shape the accompaniment's expression.",
    "midi_fn": "Play this MIDI file instead of listening to a live soloist. Leave empty to listen.",
    "init_bpm": "Tempo the accompaniment starts at, before the soloist has established one.",
    "init_velocity": "MIDI velocity the accompaniment starts at, between 1 and 127.",
    "polling_period": "Duration of one input frame, in seconds. Shorter means finer following and more CPU.",
    "adjust_following_rate": "How strongly the follower is pulled towards the expected position while the soloist is silent.",
    "expected_position_weight": "Weight of the expected position against the follower's own estimate.",
    "onset_tracker_type": "How score onsets are tracked: 'continuous' or 'discrete'.",
    "use_ceus_mediator": "Route MIDI through the CEUS mediator of a Boesendorfer CEUS piano.",
    "bypass_audio": "Send MIDI only, without rendering audio through FluidSynth.",
    "test": "Use a dummy MIDI router, so that the ACCompanion runs without any MIDI hardware.",
    "record_midi": "Record the solo input and the accompaniment output to MIDI files.",
    "accompanist_decoder_kwargs": "Extra arguments for the accompaniment decoder. Leave empty for the default.",
    "score_follower_kwargs": "The score follower that tracks the soloist's position in the score.",
    "midi_router_kwargs": "Where the ACCompanion listens and where it plays. Names are matched partially, so a fragment of a device name is enough.",
    "tempo_model_kwargs": "How the accompaniment's tempo is synchronised with the soloist's.",
    "performance_codec_kwargs": "How the accompaniment's timing, dynamics and articulation are rendered.",
    "solo_input_to_accompaniment_port_name": "MIDI input the soloist plays into.",
    "acc_output_to_sound_port_name": "MIDI output the accompaniment is played on.",
    "MIDIPlayer_to_sound_port_name": "Output for the built-in MIDI player, used in place of a live soloist.",
    "MIDIPlayer_to_accompaniment_port_name": "Port the follower listens on when the built-in MIDI player is used. Needs a virtual MIDI connection.",
    "simple_button_input_port_name": "Optional foot switch or button used to start and stop.",
    "tempo_model_kwargs.tempo_model": "Synchronisation model that decides how the accompaniment's beat period follows the soloist's.",
    "window_size": "Number of frames in the alignment window.",
    "step_size": "How far the alignment window advances per step.",
    "velocity_trend_ma_alpha": "Smoothing of the overall loudness trend, between 0 and 1. Higher follows the soloist more closely.",
    "articulation_ma_alpha": "Smoothing of articulation, between 0 and 1.",
    "velocity_dev_scale": "How strongly note-wise velocity deviations are applied.",
    "velocity_min": "Softest MIDI velocity the accompaniment plays.",
    "velocity_max": "Loudest MIDI velocity the accompaniment plays.",
    "velocity_solo_scale": "How much of the soloist's loudness carries over to the accompaniment.",
    "timing_scale": "How strongly notated timing deviations are applied.",
    "log_articulation_scale": "How strongly articulation deviations are applied.",
    "mechanical_delay": "Seconds between sending a note and the instrument sounding it. Around 0.1 to 0.2 for a Disklavier, 0 otherwise.",
    "processor": "Feature processor applied to the incoming MIDI frames.",
    "processor_kwargs": "Arguments for the feature processor, as a Python dict.",
}


def display_name(path):
    """The label shown for a parameter, given its dotted path."""
    name = path.split(".")[-1]

    if path in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[path]
    if name in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[name]

    spelled = name.replace("_", " ")

    return spelled[:1].upper() + spelled[1:]


def help_for(path):
    """The one-line explanation of a parameter, or None."""
    return _HELP.get(path, _HELP.get(path.split(".")[-1]))


# ---------------------------------------------------------------------------
# Turning the tree into a layout
# ---------------------------------------------------------------------------


def _value_widget(node, key):
    """The input element for one leaf of the configuration tree."""
    if node.type is bool:
        return sg.Checkbox(
            "",
            default=bool(node.data),
            key=key,
            background_color=BACKGROUND,
            text_color=TEXT,
            checkbox_color=FIELD,
        )
    if node.type is dict:
        return sg.Multiline(
            _format_dict(node.data),
            key=key,
            size=(FIELD_WIDTH, 3),
            font=FONT_MONO,
            background_color=FIELD,
            text_color=TEXT,
            border_width=0,
            no_scrollbar=True,
        )
    if node.type in (int, float):
        return sg.InputText(
            str(node.data),
            key=key,
            size=(14, 1),
            justification="right",
            border_width=0,
        )

    return sg.InputText(str(node.data), key=key, size=(FIELD_WIDTH, 1), border_width=0)


def _format_dict(data):
    """A dict as the one-line Python literal the user edits."""
    return "{}" if not data else repr(data)


def field_rows(node, key, widget=None):
    """A labelled row for one leaf, plus its muted caption."""
    help_text = help_for(key)

    rows = [
        [
            sg.Text(
                display_name(key),
                size=(LABEL_WIDTH, 1),
                tooltip=help_text,
                font=FONT,
            ),
            _value_widget(node, key) if widget is None else widget,
        ]
    ]

    if help_text is not None:
        rows.append(_caption(help_text))

    return rows


def child_rows(child_name, child, layout_hooks, enclosing_scope=""):
    """The rows for one child of a `ConfigurationNode`."""
    key = enclosing_scope + child_name

    layout_hook = _retrieve(key, child.type, child.data, layout_hooks)

    composite = child.type is dict and len(child.child_names_and_children) > 0

    if layout_hook is not None:
        rows = layout_hook(child, key)
    elif composite:
        rows = gui_layout(child, layout_hooks, key + ".")
    else:
        return field_rows(child, key)

    if not composite:
        # A hook that renders a whole parameter in one field -- Matchmaker's
        # method options, say -- brings its own label and caption.
        return rows

    caption = help_for(key)
    header = (
        []
        if caption is None
        else [[sg.Text(caption, font=FONT_SMALL, text_color=MUTED)]]
    )

    return [[_section(display_name(key), header + rows, nested=bool(enclosing_scope))]]


def gui_layout(config_node, layout_hooks=None, enclosing_scope=""):
    """Rows for every child of `config_node`, in signature order."""
    layout = []

    for child_name, child in config_node.child_names_and_children:
        layout.extend(child_rows(child_name, child, layout_hooks, enclosing_scope))

    return layout


# ---------------------------------------------------------------------------
# The ACCompanion variants and their score followers
# ---------------------------------------------------------------------------

#: Score followers the ACCompanion's own variants provide. Everything else
#: comes from Matchmaker and is looked up at run time, so that a follower added
#: there needs no change here. Mirrors `ACCOMPANION_SCORE_FOLLOWERS` in
#: bin/launch_acc.py.
ACCOMPANION_SCORE_FOLLOWERS = {
    "hmm": ["PitchIOIHMM", "PitchIOIKHMM"],
    "oltw": ["OnlineTimeWarping"],
}


def matchmaker_methods():
    """Every Matchmaker score follower this installation provides.

    Empty if Matchmaker is not installed, in which case the GUI offers the
    ACCompanion's own followers only.
    """
    try:
        from accompanion.score_follower.matchmaker_methods import available_methods

        return list(available_methods())
    except Exception:  # pragma: no cover -- Matchmaker missing or too old
        return []


ACCOMPANION_VARIANTS = {
    "hmm": {
        "label": "HMM  --  the ACCompanion's own hidden Markov model",
        "summary": "Note-level following against a hidden Markov model of the "
        "solo part. The default, and a good fit for the simple pieces.",
        "module": "accompanion.hmm_accompanion",
        "class_name": "HMMACCompanion",
        "score_followers": lambda: list(ACCOMPANION_SCORE_FOLLOWERS["hmm"]),
    },
    "oltw": {
        "label": "OLTW  --  on-line time warping",
        "summary": "Frame-level following by on-line dynamic time warping. "
        "Usually used for the four-hand pieces.",
        "module": "accompanion.oltw_accompanion",
        "class_name": "OLTWACCompanion",
        "score_followers": lambda: list(ACCOMPANION_SCORE_FOLLOWERS["oltw"]),
    },
    "matchmaker": {
        "label": "Matchmaker  --  any follower the Matchmaker package provides",
        "summary": "Runs a score follower from Matchmaker's registry, chosen "
        "by name. The list below is read live from the installed Matchmaker.",
        "module": "accompanion.matchmaker_accompanion",
        "class_name": "MatchmakerACCompanion",
        "score_followers": matchmaker_methods,
    },
}

#: What each variant's `score_follower_kwargs` looks like. Mirrors
#: `SCORE_FOLLOWER_DEFAULTS` in bin/launch_acc.py.
_SCORE_FOLLOWER_DEFAULTS = {
    "hmm": {
        "score_follower": "PitchIOIHMM",
        "input_processor": {
            "processor": "PitchIOIProcessor",
            "processor_kwargs": {"piano_range": True},
        },
    },
    "oltw": {
        "score_follower": "OnlineTimeWarping",
        "window_size": 100,
        "step_size": 10,
        "input_processor": {
            "processor": "PianoRollProcessor",
            "processor_kwargs": {"piano_range": True},
        },
    },
    "matchmaker": {
        "score_follower": "hmm",
        "score_follower_kwargs": {},
    },
}


def score_follower_defaults(variant_key, method=None):
    """`score_follower_kwargs` for a variant, following `method`."""
    from copy import deepcopy

    defaults = deepcopy(_SCORE_FOLLOWER_DEFAULTS[variant_key])

    if method:
        defaults["score_follower"] = method

    return defaults


def follower_description(variant_key, method):
    """What the chosen score follower is, and how it wants to be fed.

    For a Matchmaker method this is read out of Matchmaker's own spec, so it
    stays true for methods this file has never heard of.
    """
    if not method:
        return "No score follower available."

    if variant_key != "matchmaker":
        blurbs = {
            "PitchIOIHMM": "Hidden Markov model over pitch and inter-onset "
            "intervals, with the ACCompanion's own tempo model.",
            "PitchIOIKHMM": "As PitchIOIHMM, with a Kalman filter smoothing "
            "the tempo estimate.",
            "OnlineTimeWarping": "On-line dynamic time warping against a "
            "piano-roll rendering of the score.",
        }

        return blurbs.get(method, f"The ACCompanion's '{method}' follower.")

    try:
        from accompanion.score_follower.matchmaker_methods import (
            method_defaults,
            preferred_polling_period,
        )
    except Exception:  # pragma: no cover -- Matchmaker missing
        return f"Matchmaker method '{method}'."

    if method not in matchmaker_methods():
        # A method a configuration file names that the installed Matchmaker
        # does not register. It is still offered, so that loading a file does
        # not quietly swap the follower out from under the user.
        return (
            f"'{method}' is not registered by the Matchmaker installed here, "
            "so starting will fail. Install the version that provides it, or "
            "pick another method."
        )

    defaults = method_defaults(method)
    period = preferred_polling_period(method)
    processor = defaults.pop("processor", "the default")
    defaults.pop("polling_period", None)

    if period is None:
        feed = "aligns note by note, so it is fed one MIDI message at a time"
    else:
        feed = f"fed frames of {period:g} s"

    spelled = ", ".join(
        f"{k}={getattr(v, '__name__', v)}" for k, v in defaults.items()
    )

    return (
        f"Matchmaker '{method}': feature processor '{processor}', {feed}."
        + (f"  Defaults: {spelled}." if spelled else "")
    )


# ---------------------------------------------------------------------------
# Hook: MIDI ports
# ---------------------------------------------------------------------------

#: Shown in the port drop-downs for a port that is deliberately not used.
NO_PORT = "(none)"

RESCAN_KEY = "-RESCAN-PORTS-"


def _port_directions():
    """``{port parameter: True if it is an input}``, in `MidiRouter`'s order."""
    router = load_class("accompanion.midi_handler.midi_routing", "MidiRouter")

    inputs = {
        "solo_input_to_accompaniment_port_name",
        "simple_button_input_port_name",
    }

    return {
        p.name: (p.name in inputs or "input" in p.name)
        for p in class_init_args(router.__init__)
    }


def _available_ports():
    """``{port parameter: [port names]}`` for the ports plugged in right now."""
    from mido import get_input_names, get_output_names

    in_ports = get_input_names()
    out_ports = get_output_names()

    return {
        name: list(in_ports) if is_input else list(out_ports)
        for name, is_input in _port_directions().items()
    }


def midi_router_kwargs_trigger(name, data_type, data):
    return name == "midi_router_kwargs"


def midi_router_kwargs_configuration(value):
    """One string node per port, defaulting to what is plugged in."""
    assert (
        type(value) is dict
    ), "midi_router_kwargs_configuration was expected to be a dict"

    ports = _available_ports()

    #: The two ports the ACCompanion cannot run without are pointed at the
    #: first device found; the rest stay off until the user picks one.
    essential = (
        "solo_input_to_accompaniment_port_name",
        "acc_output_to_sound_port_name",
    )

    child_names_and_children = []

    for port_name, available in ports.items():
        data = value.get(port_name, "")

        if isinstance(data, int) and not isinstance(data, bool):
            # A port given by index, as the older configuration files spell
            # it. The drop-down deals in names, so resolve it to one.
            data = available[data] if 0 <= data < len(available) else ""

        if data in (None, "", "None", "none", NO_PORT):
            data = available[0] if port_name in essential and available else ""

        child_names_and_children.append(
            (port_name, ConfigurationNode(str, data=str(data)))
        )

    return ConfigurationNode(dict, child_names_and_children=child_names_and_children)


def midi_router_kwargs_layout(config_node, enclosing_scope):
    ports = _available_ports()

    rows = []

    for port_name, child in config_node.child_names_and_children:
        key = enclosing_scope + "." + port_name
        available = ports.get(port_name, [])
        choices = [NO_PORT] + available

        if child.data and child.data not in choices:
            # A port named in the configuration that is not plugged in right
            # now. It is offered anyway: port names are matched partially at
            # run time, and the device may well be connected before the piece
            # starts.
            choices.insert(1, child.data)

        rows.extend(
            field_rows(
                child,
                key,
                widget=_combo(choices, child.data or NO_PORT, key),
            )
        )

    rows.append(
        [
            sg.Text("", size=(LABEL_WIDTH, 1)),
            secondary_button("Rescan MIDI ports", RESCAN_KEY),
        ]
    )

    return rows


def midi_port_rescan_trigger(name, data_type, data):
    return name == RESCAN_KEY


def midi_port_rescan_update(window, event, values):
    """Re-read the connected devices into every port drop-down."""
    ports = _available_ports()

    for key, element in window.key_dict.items():
        if not isinstance(key, str) or not key.startswith("midi_router_kwargs."):
            continue

        available = ports.get(key.split(".")[-1])

        if available is None:
            continue

        current = values.get(key, NO_PORT)
        choices = [NO_PORT] + available

        if current not in choices and current != NO_PORT:
            choices.insert(1, current)

        _update_combo(element, values=choices, value=current)

    found = sorted({p for names in ports.values() for p in names})

    return f"{len(found)} MIDI port(s) found." if found else "No MIDI ports found."


# ---------------------------------------------------------------------------
# Hook: tempo model
# ---------------------------------------------------------------------------


def _sync_model_names():
    import accompanion.accompanist.tempo_models as tempo_models

    # `SyncModel` itself is the abstract base and cannot be instantiated.
    return sorted(
        name
        for name in dir(tempo_models)
        if name.endswith("SyncModel") and name != "SyncModel"
    )


def tempo_model_trigger(name, data_type, data):
    return name.split(".")[-1] == "tempo_model"


def tempo_model_configuration(value):
    """The model's class name, resolving the short aliases such as ``LSM``."""
    import accompanion.accompanist.tempo_models as tempo_models

    name = value.__name__ if isinstance(value, type) else str(value)

    if name not in _sync_model_names():
        resolved = getattr(tempo_models, name, None)
        name = resolved.__name__ if isinstance(resolved, type) else ""

    return ConfigurationNode(str, data=name)


def tempo_model_layout(config_node, enclosing_scope):
    names = _sync_model_names()

    return field_rows(
        config_node,
        enclosing_scope,
        widget=_combo(names, config_node.data or (names[0] if names else ""), enclosing_scope),
    )


# ---------------------------------------------------------------------------
# Hook: score follower
# ---------------------------------------------------------------------------

FOLLOWER_INFO_KEY = "-FOLLOWER-INFO-"

_FOLLOWER_KEY = "score_follower_kwargs.score_follower"
_FOLLOWER_OPTIONS_KEY = "score_follower_kwargs.score_follower_kwargs"


def score_follower_hooks(variant_key):
    """Hooks rendering the score follower panel for one ACCompanion variant."""
    followers = ACCOMPANION_VARIANTS[variant_key]["score_followers"]()

    def method_layout(config_node, enclosing_scope):
        current = config_node.data or (followers[0] if followers else "")
        choices = list(followers)

        if current and current not in choices:
            # Whatever the configuration named stays selected even if this
            # installation does not provide it; `follower_description` says so,
            # rather than the GUI silently choosing something else.
            choices.insert(0, current)

        return [
            [
                sg.Text(
                    display_name(enclosing_scope),
                    size=(LABEL_WIDTH, 1),
                    tooltip="The score follower that tracks the soloist.",
                ),
                _combo(choices, current, enclosing_scope, enable_events=True),
            ],
            _caption(
                follower_description(variant_key, current), key=FOLLOWER_INFO_KEY
            ),
        ]

    def method_changed(window, event, values):
        method = values[event]

        window[FOLLOWER_INFO_KEY].update(follower_description(variant_key, method))

        if "-SUBTITLE-" in window.key_dict:
            window["-SUBTITLE-"].update(_subtitle(variant_key, method))

        if variant_key == "matchmaker" and _FOLLOWER_OPTIONS_KEY in window.key_dict:
            # The options a method accepts are its own, so carrying them over
            # to another method would hand it keys it does not declare.
            window[_FOLLOWER_OPTIONS_KEY].update("{}")

        return f"Score follower set to '{method}'."

    hooks = [
        Hook(
            lambda n, t, d: n == _FOLLOWER_KEY,
            layout=method_layout,
        ),
        Hook(
            lambda n, t, d: n == _FOLLOWER_KEY,
            update=method_changed,
        ),
    ]

    if variant_key == "matchmaker":
        hooks.extend(
            [
                Hook(
                    lambda n, t, d: n == _FOLLOWER_OPTIONS_KEY,
                    # Matchmaker's `kwargs` dict is edited as one literal, not
                    # split into fields: which keys are valid depends on the
                    # method, and a composite method such as the ensemble
                    # nests a list of members in here.
                    configuration=lambda value: ConfigurationNode(
                        dict, data=dict(value) if isinstance(value, dict) else {}
                    ),
                ),
                Hook(
                    lambda n, t, d: n == _FOLLOWER_OPTIONS_KEY,
                    layout=_follower_options_layout,
                ),
            ]
        )

    return hooks


def _follower_options_layout(config_node, enclosing_scope):
    return [
        [
            sg.Text(
                display_name(enclosing_scope),
                size=(LABEL_WIDTH, 1),
                tooltip="Matchmaker's kwargs for the chosen method.",
            ),
            sg.Multiline(
                _format_dict(config_node.data),
                key=enclosing_scope,
                size=(FIELD_WIDTH, 4),
                font=FONT_MONO,
                background_color=FIELD,
                text_color=TEXT,
                border_width=0,
                no_scrollbar=True,
            ),
        ],
        _caption(
            "A Python dict, merged over the method's own defaults. The keys a "
            "method accepts are the ones it declares in Matchmaker's "
            "methods.yaml, so leave this empty after switching methods."
        ),
    ]


# ---------------------------------------------------------------------------
# Hook: file names
# ---------------------------------------------------------------------------

_FILE_TYPES = {
    "solo_fn": (
        ("Scores", "*.musicxml *.mxl *.xml"),
        ("Match files", "*.match"),
        ("All files", "*.*"),
    ),
    "acc_fn": (
        ("Scores", "*.musicxml *.mxl *.xml"),
        ("All files", "*.*"),
    ),
    "accompaniment_match": (("Match files", "*.match"), ("All files", "*.*")),
    "midi_fn": (("MIDI files", "*.mid *.midi"), ("All files", "*.*")),
}


def single_file_name_trigger(name, data_type, data):
    last = name.split(".")[-1]

    return (last.endswith("_fn") or last in _FILE_TYPES) and data_type is str


def _pieces_folder():
    for candidate in (
        os.path.join(_repo_root(), "accompanion_pieces", "simple_pieces"),
        os.path.join(_repo_root(), "sample_pieces"),
    ):
        if os.path.isdir(candidate):
            return candidate

    return _repo_root()


def single_file_name_layout(config_node, enclosing_scope):
    name = enclosing_scope.split(".")[-1]

    return field_rows(
        config_node,
        enclosing_scope,
        widget=sg.Column(
            [
                [
                    sg.InputText(
                        config_node.data,
                        key=enclosing_scope,
                        size=(FIELD_WIDTH, 1),
                        border_width=0,
                    ),
                    sg.FileBrowse(
                        "Browse",
                        target=enclosing_scope,
                        key=enclosing_scope + "_filebrowse",
                        initial_folder=_pieces_folder(),
                        file_types=_FILE_TYPES.get(
                            name, (("All files", "*.*"),)
                        ),
                        font=FONT,
                        button_color=(TEXT, FIELD),
                    ),
                ]
            ],
            pad=(0, 0),
        ),
    )


def multiple_file_name_trigger(name, data_type, data):
    return name.split(".")[-1].endswith("_fn") and data_type in (list,)


def multiple_file_name_layout(config_node, enclosing_scope):
    return field_rows(
        config_node,
        enclosing_scope,
        widget=sg.Column(
            [
                [
                    sg.Multiline(
                        "\n".join(config_node.data),
                        autoscroll=True,
                        key=enclosing_scope,
                        enable_events=True,
                        size=(FIELD_WIDTH, 4),
                        font=FONT_MONO,
                        background_color=FIELD,
                        text_color=TEXT,
                        border_width=0,
                    ),
                    sg.FilesBrowse(
                        "Browse",
                        enable_events=True,
                        key=enclosing_scope + "_browse",
                        target=enclosing_scope + "_browse",
                        files_delimiter="\n",
                        initial_folder=_pieces_folder(),
                        font=FONT,
                        button_color=(TEXT, FIELD),
                    ),
                ]
            ],
            pad=(0, 0),
        ),
    )


def multiple_file_name_eval(config_string):
    return [line for line in config_string.split("\n") if line.strip()]


def multiple_file_name_browse_trigger(name, data_type, data):
    return name.split(".")[-1].endswith("_fn_browse")


def multiple_file_name_browse_update(window, event, values):
    file_names = [name for name in values[event].split("\n") if name.strip()]

    if len(file_names) > 0:
        window[event[: -len("_browse")]].update("\n".join(file_names))

    return f"{len(file_names)} file(s) selected."


# ---------------------------------------------------------------------------
# Hooks: plain values
# ---------------------------------------------------------------------------

#: Parameters the ACCompanion treats as flags even where their annotation is
#: looser than `bool` -- `record_midi` is `Optional[str]` on `HMMACCompanion`
#: and `bool` on the others.
_FLAGS = ("test", "bypass_audio", "use_ceus_mediator", "record_midi")


def flag_trigger(name, data_type, data):
    return data_type is bool or name.split(".")[-1] in _FLAGS


def flag_configuration(value):
    return ConfigurationNode(bool, data=bool(value))


def flag_evaluation(config_string):
    if isinstance(config_string, bool):
        return config_string

    return str(config_string).strip().lower() in ("true", "1", "yes", "on")


def dict_leaf_trigger(name, data_type, data):
    return data_type is dict


def dict_leaf_evaluation(config_string):
    """A dict typed as a Python literal; blank means an empty dict."""
    text = str(config_string).strip()

    if not text:
        return {}

    value = literal_eval(text)

    if not isinstance(value, dict):
        raise ValueError(f"expected a dict, got {type(value).__name__}")

    return value


# ---------------------------------------------------------------------------
# Pieces and configuration files
# ---------------------------------------------------------------------------

PIECE_KEY = "-PIECE-"
PIECE_PROMPT = "Choose a piece..."

STATUS_KEY = "-STATUS-"
SAVE_KEY = "-SAVE-"
LOAD_KEY = "-LOAD-"
RESET_KEY = "-RESET-"
START_KEY = "-START-"
CANCEL_KEY = "-CANCEL-"

#: Where the repository keeps its pieces, relative to its root.
_PIECE_FOLDERS = (
    os.path.join("accompanion_pieces", "simple_pieces"),
    os.path.join("accompanion_pieces", "complex_pieces"),
    "sample_pieces",
)


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def discover_pieces():
    """``{label: (solo_fn, acc_fn)}`` for every piece already split in two.

    A piece is listed once it holds a ``primo.musicxml`` and a
    ``secondo.musicxml``; the pieces that still have to be split into a solo
    and an accompaniment part are left out, since the GUI cannot split them.
    """
    pieces = {}

    for folder in _PIECE_FOLDERS:
        root = os.path.join(_repo_root(), folder)

        if not os.path.isdir(root):
            continue

        for name in sorted(os.listdir(root)):
            solo_fn = os.path.join(root, name, "primo.musicxml")
            acc_fn = os.path.join(root, name, "secondo.musicxml")

            if os.path.isfile(solo_fn) and os.path.isfile(acc_fn):
                pieces[f"{name}   ({folder.replace(os.sep, '/')})"] = (solo_fn, acc_fn)

    return pieces


def piece_hook(pieces):
    """Hook filling in the two scores when a piece is picked from the list."""

    def piece_chosen(window, event, values):
        chosen = pieces.get(values[event])

        if chosen is None:
            return None

        solo_fn, acc_fn = chosen
        window["solo_fn"].update(solo_fn)
        window["acc_fn"].update(acc_fn)

        return f"Loaded the scores of {values[event].split('   ')[0]}."

    return Hook(lambda n, t, d: n == PIECE_KEY, update=piece_chosen)


def piece_rows(pieces):
    """The piece picker, shown above the individual score fields."""
    if not pieces:
        return []

    return [
        [
            sg.Text("Piece", size=(LABEL_WIDTH, 1)),
            _combo(
                [PIECE_PROMPT] + list(pieces), PIECE_PROMPT, PIECE_KEY,
                enable_events=True,
            ),
        ],
        _caption(
            "Fills in the two scores below. Pick the files by hand for a piece "
            "that is not listed here."
        ),
        [sg.HorizontalSeparator(color=BORDER, pad=((0, 0), (10, 12)))],
    ]


def _piece_dir_of(solo_fn):
    """The piece folder a score sits in, if it is one of the repository's."""
    if not isinstance(solo_fn, str) or not solo_fn:
        return None

    piece_dir = os.path.dirname(os.path.abspath(solo_fn))
    parent = os.path.dirname(piece_dir)

    for folder in _PIECE_FOLDERS:
        if os.path.abspath(os.path.join(_repo_root(), folder)) == parent:
            return os.path.basename(piece_dir)

    return None


def load_configuration_file(path):
    """Read a configuration file into ``(config dict, follower name)``.

    Understands the files in ``config_files/`` -- a ``config`` mapping beside a
    ``piece_dir`` naming a folder under ``accompanion_pieces`` -- as well as
    the flat files this GUI writes.
    """
    with open(path, "r") as config_file:
        try:
            info = yaml.safe_load(config_file)
        except yaml.YAMLError:
            # Files written by older versions of this GUI carry Python class
            # references, which only the unsafe loader resolves. The file was
            # picked by the user, so it is as trusted as the GUI itself.
            config_file.seek(0)
            info = yaml.unsafe_load(config_file)

    if not isinstance(info, dict):
        raise ValueError("a configuration file has to hold a mapping")

    config = dict(info.get("config", info))
    follower = config.pop("follower", info.get("follower"))

    _resolve_piece_dir(info, config)

    for key, value in list(config.items()):
        # YAML has no None literal spelled 'None', so the port names in the
        # repository's files arrive as that string.
        if value == "None":
            config[key] = None

    return config, follower


def _resolve_piece_dir(info, config):
    """Turn a ``piece_dir`` into the score paths the constructor wants."""
    piece = info.get("piece_dir")

    if not piece:
        return

    for folder in _PIECE_FOLDERS:
        piece_dir = os.path.join(_repo_root(), folder, piece)

        if not os.path.isdir(piece_dir):
            continue

        for name in ("solo_fn", "acc_fn", "accompaniment_match", "midi_fn"):
            relative = info.get(name)

            if relative:
                config[name] = os.path.join(piece_dir, os.path.normpath(relative))

        for name, default in (
            ("solo_fn", "primo.musicxml"),
            ("acc_fn", "secondo.musicxml"),
        ):
            candidate = os.path.join(piece_dir, default)

            if not config.get(name) and os.path.isfile(candidate):
                config[name] = candidate

        if not config.get("solo_fn"):
            # The four-hand pieces follow a set of recorded performances of
            # the solo part rather than its score, as bin/launch_acc.py does.
            import glob

            recordings = sorted(
                glob.glob(os.path.join(piece_dir, "match", "cc_solo", "*.match"))
            )

            if recordings:
                config["solo_fn"] = recordings[-5:]

        return


def save_configuration_file(path, config, variant_key):
    """Write a configuration in the shape ``config_files/`` uses."""
    document = {"config": dict(config, follower=variant_key)}

    piece = _piece_dir_of(config.get("solo_fn"))

    if piece is not None:
        document["piece_dir"] = piece

    with open(path, "w") as destination:
        destination.write(
            "# ACCompanion configuration, written by the configuration GUI.\n"
        )
        yaml.safe_dump(document, destination, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Collecting the configuration
# ---------------------------------------------------------------------------


def _create_config(values, config_tree, evaluation_hooks, type_checked):
    """Evaluate the widget contents into a config dict.

    Returns ``(config, error message)``, exactly one of which is None.
    """
    for k in values.keys():
        if not isinstance(k, str):
            continue

        result = config_tree.search(k)

        if result is None or len(result.child_names_and_children) > 0:
            continue

        evaluate = _retrieve(k, result.type, result.data, evaluation_hooks)
        raw = values[k]

        try:
            if evaluate is not None:
                result.data = evaluate(raw)
            elif result.type is str:
                result.data = str(raw)
            elif result.type in (int, float):
                result.data = _to_number(raw, result.type)
            else:
                result.data = literal_eval(str(raw).strip())
        except (ValueError, SyntaxError, TypeError) as error:
            return None, f"{display_name(k)}: {error}"

    if type_checked:
        try:
            check_for_type_error(config_tree)
        except TypeError as error:
            return None, str(error).replace("\n", " ")

    return _normalise_config(config_tree.value()), None


def _to_number(raw, number_type):
    """A number typed into a field, or a complaint a musician can read."""
    text = str(raw).strip()

    try:
        return number_type(float(text))
    except (TypeError, ValueError):
        raise ValueError(f"'{text}' is not a number") from None


def _normalise_config(config):
    """Spell 'not set' the way the ACCompanion's constructors expect it."""
    ports = config.get("midi_router_kwargs")

    if isinstance(ports, dict):
        config["midi_router_kwargs"] = {
            name: (None if value in ("", NO_PORT) else value)
            for name, value in ports.items()
        }

    for name in ("midi_fn", "accompaniment_match"):
        if config.get(name) == "":
            config[name] = None

    if config.get("accompanist_decoder_kwargs") == {}:
        config["accompanist_decoder_kwargs"] = None

    return config


def _validate_config(config):
    """A message naming the first thing that would stop a run, or None."""
    for name in ("solo_fn", "acc_fn"):
        value = config.get(name)
        paths = value if isinstance(value, (list, tuple)) else [value]

        if not value or not all(paths):
            return f"{display_name(name)} is not set."

        for path in paths:
            if not os.path.isfile(path):
                return f"{display_name(name)}: no such file '{path}'."

    ports = config.get("midi_router_kwargs") or {}

    if not config.get("test") and not ports.get(
        "solo_input_to_accompaniment_port_name"
    ) and not config.get("midi_fn"):
        return (
            "No solo input port is set. Choose one under MIDI, or play a MIDI "
            "file instead, or switch on dummy MIDI routing."
        )

    return None


class ConfigurationReload(object):
    """Asks the caller to build the configuration window again.

    Returned when the user loads a configuration file or resets the form: the
    file may name a different ACCompanion than the one on screen, so the window
    has to be generated again from that class' signature.
    """

    __slots__ = ("config", "follower")

    def __init__(self, config=None, follower=None):
        self.config = config
        self.follower = follower


# ---------------------------------------------------------------------------
# The configuration window
# ---------------------------------------------------------------------------

#: Which tab each constructor parameter belongs on. The last group takes
#: whatever is left, so a parameter added to a constructor still shows up.
_PARAMETER_GROUPS = (
    ("Piece", ("solo_fn", "acc_fn", "accompaniment_match", "midi_fn")),
    ("Score follower", ("score_follower_kwargs",)),
    (
        "MIDI",
        ("midi_router_kwargs", "test", "record_midi", "bypass_audio", "use_ceus_mediator"),
    ),
    (
        "Performance",
        ("init_bpm", "init_velocity", "tempo_model_kwargs", "performance_codec_kwargs"),
    ),
    ("Advanced", None),
)


def _tab(title, rows):
    return sg.Tab(
        f"   {title}   ",
        [
            [
                sg.Column(
                    rows,
                    scrollable=True,
                    vertical_scroll_only=True,
                    size=_TAB_SIZE,
                    pad=((16, 16), (14, 14)),
                    background_color=BACKGROUND,
                    expand_x=True,
                )
            ]
        ],
        background_color=BACKGROUND,
    )


def _tabs(config_tree, layout_hooks, extra_rows=None):
    """One tab per group of parameters, in the order `_PARAMETER_GROUPS` sets."""
    extra_rows = extra_rows or {}

    rows_by_name = {
        child_name: child_rows(child_name, child, layout_hooks)
        for child_name, child in config_tree.child_names_and_children
    }

    tabs = []
    placed = set()

    for title, names in _PARAMETER_GROUPS:
        if names is None:
            members = [name for name in rows_by_name if name not in placed]
        else:
            members = [name for name in names if name in rows_by_name]

        placed.update(members)

        rows = list(extra_rows.get(title, []))

        for name in members:
            rows.extend(rows_by_name[name])

        if rows:
            tabs.append(_tab(title, rows))

    return tabs


def class_init_configurations_via_gui(
    class_object,
    window_title=None,
    subtitle=None,
    hooks=(),
    defaults=None,
    extra_rows=None,
    save_tag=None,
    type_checked=True,
):
    """Configure a class' constructor arguments in a window.

    Parameters
    ----------
    class_object : type
        The class whose constructor is being configured.
    window_title, subtitle : str, optional
        Shown in the title bar and under the heading.
    hooks : iterable of Hook
        Overrides for how individual parameters are built, laid out, updated
        and evaluated.
    defaults : dict, optional
        The values the fields start at. `constructor_defaults` by default.
    extra_rows : dict, optional
        ``{tab title: rows}``, inserted at the top of that tab. Used for the
        piece picker, which is not a constructor parameter.
    save_tag : str, optional
        Written as ``follower`` into a saved configuration file.
    type_checked : bool
        Whether the collected values are checked against the parameter types
        before they are handed back.

    Returns
    -------
    dict or ConfigurationReload or None
        The configuration, a request to build the window again, or None if the
        user closed or cancelled the window.
    """
    apply_theme()

    underlying_dict = constructor_defaults(class_object) if defaults is None else defaults

    hook_init_args = [
        p.name for p in class_init_args(Hook.__init__) if p.name != "trigger"
    ]

    hook_system = {name: _no_hooks() for name in hook_init_args}

    for hook in hooks:
        for name in hook_init_args:
            function = getattr(hook, name, None)

            if not function is None:
                hook_system[name]["triggers"].append(hook.trigger)
                hook_system[name]["functions"].append(function)

    if window_title is None:
        window_title = class_object.__name__ + " configuration"

    config_tree = configuration_tree(underlying_dict, hook_system["configuration"])

    layout = [
        [
            sg.Column(
                [
                    [sg.Text("ACCompanion", font=FONT_TITLE, text_color=ACCENT)],
                    [
                        sg.Text(
                            subtitle or class_object.__name__,
                            font=FONT_SMALL,
                            text_color=MUTED,
                            size=(80, 1),
                            key="-SUBTITLE-",
                        )
                    ],
                ],
                pad=((16, 16), (14, 10)),
                expand_x=True,
            )
        ],
        [sg.HorizontalSeparator(color=BORDER)],
        [
            sg.TabGroup(
                [_tabs(config_tree, hook_system["layout"], extra_rows)],
                font=FONT_BOLD,
                background_color=BACKGROUND,
                tab_background_color=BACKGROUND,
                selected_background_color=FIELD,
                title_color=MUTED,
                selected_title_color=ACCENT,
                border_width=0,
                tab_border_width=0,
                pad=((16, 16), (12, 6)),
                expand_x=True,
            )
        ],
        [sg.HorizontalSeparator(color=BORDER)],
        [
            sg.Column(
                [
                    [
                        sg.Text(
                            "",
                            key=STATUS_KEY,
                            size=(58, 2),
                            font=FONT_SMALL,
                            text_color=MUTED,
                        ),
                        sg.Push(),
                        secondary_button("Reset", RESET_KEY),
                        secondary_button("Load...", LOAD_KEY),
                        secondary_button("Save as...", SAVE_KEY),
                        secondary_button("Cancel", CANCEL_KEY),
                        primary_button("Start ACCompanion", START_KEY),
                    ]
                ],
                pad=((16, 16), (10, 14)),
                expand_x=True,
            )
        ],
    ]

    window = sg.Window(window_title, layout, finalize=True, margins=(0, 0))
    _restyle_combos(window)

    def status(message, colour=MUTED):
        window[STATUS_KEY].update(message, text_color=colour)

    try:
        while True:
            event, values = window.read()

            if event in (sg.WINDOW_CLOSED, CANCEL_KEY):
                return None

            update = (
                _retrieve(event, str, values, hook_system["update"])
                if isinstance(event, str)
                else None
            )

            if update is not None:
                message = update(window, event, values)

                if message:
                    status(message)
                continue

            if event == RESET_KEY:
                return ConfigurationReload(config=None, follower=save_tag)

            if event == LOAD_KEY:
                path = sg.popup_get_file(
                    "Load a configuration",
                    no_window=True,
                    initial_folder=os.path.join(_repo_root(), "config_files"),
                    file_types=(("Configurations", "*.yml *.yaml"), ("All files", "*.*")),
                )

                if not path:
                    continue

                try:
                    config, follower = load_configuration_file(path)
                except (OSError, ValueError, yaml.YAMLError) as error:
                    status(f"Could not read {os.path.basename(path)}: {error}", DANGER)
                    continue

                return ConfigurationReload(config=config, follower=follower or save_tag)

            if event in (SAVE_KEY, START_KEY):
                config, error = _create_config(
                    values, config_tree, hook_system["evaluation"], type_checked
                )

                if config is None:
                    status(error, DANGER)
                    continue

                if event == START_KEY:
                    complaint = _validate_config(config)

                    if complaint is not None:
                        status(complaint, DANGER)
                        continue

                    return config

                path = sg.popup_get_file(
                    "Save the configuration",
                    save_as=True,
                    no_window=True,
                    default_extension=".yml",
                    initial_folder=_gui_config_dir(),
                    file_types=(("Configurations", "*.yml *.yaml"),),
                )

                if not path:
                    continue

                try:
                    save_configuration_file(path, config, save_tag)
                except (OSError, yaml.YAMLError) as error:
                    status(f"Could not save: {error}", DANGER)
                else:
                    status(f"Saved to {path}", SUCCESS)
    finally:
        window.close()


def _gui_config_dir():
    """Where configurations saved from the GUI go, created on first use."""
    path = os.path.join(_repo_root(), "gui_config_files")
    os.makedirs(path, exist_ok=True)

    return path


# ---------------------------------------------------------------------------
# The chooser window
# ---------------------------------------------------------------------------

METHOD_KEY = "-METHOD-"
METHOD_INFO_KEY = "-METHOD-INFO-"
CONTINUE_KEY = "-CONTINUE-"
BROWSE_KEY = "-BROWSE-CONFIG-"

_DEFAULT_VARIANT = "hmm"


def _variant_of(values):
    for key in ACCOMPANION_VARIANTS:
        if values.get(f"-VARIANT-{key}-"):
            return key

    return _DEFAULT_VARIANT


def choose_variant_via_gui():
    """Ask which ACCompanion to run, and with which score follower.

    Returns ``{"variant": key, "method": name}``, ``{"path": path}`` if the
    user chose to start from a configuration file, or None if they cancelled.
    """
    apply_theme()

    followers = {
        key: variant["score_followers"]()
        for key, variant in ACCOMPANION_VARIANTS.items()
    }

    rows = []

    for key, variant in ACCOMPANION_VARIANTS.items():
        available = len(followers[key]) > 0

        rows.append(
            [
                sg.Radio(
                    variant["label"],
                    "VARIANT",
                    default=(key == _DEFAULT_VARIANT),
                    key=f"-VARIANT-{key}-",
                    enable_events=True,
                    disabled=not available,
                    font=FONT_BOLD,
                    text_color=TEXT if available else MUTED,
                    circle_color=FIELD,
                )
            ]
        )
        rows.append(
            [
                sg.Text("", size=(3, 1)),
                sg.Text(
                    variant["summary"]
                    if available
                    else "Not available: Matchmaker is not installed in this "
                    "environment.",
                    size=(72, 2),
                    font=FONT_SMALL,
                    text_color=MUTED,
                ),
            ]
        )

    start = followers[_DEFAULT_VARIANT]

    layout = [
        [
            sg.Column(
                [
                    [sg.Text("ACCompanion", font=FONT_TITLE, text_color=ACCENT)],
                    [
                        sg.Text(
                            "An expressive accompaniment system.",
                            font=FONT_SMALL,
                            text_color=MUTED,
                        )
                    ],
                ],
                pad=((20, 20), (18, 12)),
                expand_x=True,
            )
        ],
        [sg.HorizontalSeparator(color=BORDER)],
        [
            sg.Column(
                [[sg.Text("Which follower should listen to the soloist?", font=FONT_BOLD)]]
                + rows
                + [
                    [sg.HorizontalSeparator(color=BORDER, pad=((0, 0), (12, 12)))],
                    [
                        sg.Text("Score follower", size=(16, 1)),
                        _combo(
                            start,
                            start[0] if start else "",
                            METHOD_KEY,
                            width=44,
                            enable_events=True,
                        ),
                    ],
                    [
                        sg.Text("", size=(16, 1)),
                        sg.Text(
                            follower_description(
                                _DEFAULT_VARIANT, start[0] if start else ""
                            ),
                            key=METHOD_INFO_KEY,
                            size=(60, 3),
                            font=FONT_SMALL,
                            text_color=MUTED,
                        ),
                    ],
                ],
                pad=((20, 20), (14, 6)),
                expand_x=True,
            )
        ],
        [sg.HorizontalSeparator(color=BORDER)],
        [
            sg.Column(
                [
                    [
                        secondary_button("Start from a configuration file...", BROWSE_KEY),
                        sg.Push(),
                        secondary_button("Cancel", CANCEL_KEY),
                        primary_button("Continue", CONTINUE_KEY),
                    ]
                ],
                pad=((20, 20), (12, 16)),
                expand_x=True,
            )
        ],
    ]

    window = sg.Window("ACCompanion", layout, finalize=True, margins=(0, 0))
    _restyle_combos(window)

    try:
        while True:
            event, values = window.read()

            if event in (sg.WINDOW_CLOSED, CANCEL_KEY):
                return None

            if isinstance(event, str) and event.startswith("-VARIANT-"):
                variant_key = _variant_of(values)
                choices = followers[variant_key]
                _update_combo(
                    window[METHOD_KEY],
                    values=choices,
                    value=choices[0] if choices else "",
                )
                window[METHOD_INFO_KEY].update(
                    follower_description(variant_key, choices[0] if choices else "")
                )
            elif event == METHOD_KEY:
                window[METHOD_INFO_KEY].update(
                    follower_description(_variant_of(values), values[METHOD_KEY])
                )
            elif event == BROWSE_KEY:
                path = sg.popup_get_file(
                    "Load a configuration",
                    no_window=True,
                    initial_folder=os.path.join(_repo_root(), "config_files"),
                    file_types=(("Configurations", "*.yml *.yaml"), ("All files", "*.*")),
                )

                if path:
                    return {"path": path}
            elif event == CONTINUE_KEY:
                return {"variant": _variant_of(values), "method": values[METHOD_KEY]}
    finally:
        window.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def accompanion_hooks(variant_key, pieces):
    """Every Hook the ACCompanion's configuration window uses."""
    return tuple(score_follower_hooks(variant_key)) + (
        Hook(
            midi_router_kwargs_trigger,
            layout=midi_router_kwargs_layout,
            configuration=midi_router_kwargs_configuration,
        ),
        Hook(midi_port_rescan_trigger, update=midi_port_rescan_update),
        Hook(
            tempo_model_trigger,
            layout=tempo_model_layout,
            configuration=tempo_model_configuration,
        ),
        Hook(single_file_name_trigger, layout=single_file_name_layout),
        Hook(
            multiple_file_name_trigger,
            layout=multiple_file_name_layout,
            evaluation=multiple_file_name_eval,
        ),
        Hook(multiple_file_name_browse_trigger, update=multiple_file_name_browse_update),
        Hook(
            flag_trigger,
            configuration=flag_configuration,
            evaluation=flag_evaluation,
        ),
        Hook(dict_leaf_trigger, evaluation=dict_leaf_evaluation),
        piece_hook(pieces),
    )


def accompanion_configurations_and_version_via_gui():
    """Configure an ACCompanion in the GUI.

    Returns ``(configurations, ACCompanion class)``, or ``(None, None)`` if the
    user aborted. The configurations are the keyword arguments of the returned
    class' constructor.
    """
    apply_theme()

    choice = choose_variant_via_gui()

    if choice is None:
        print("Configuration aborted")
        return None, None

    pieces = discover_pieces()

    variant_key = choice.get("variant", _DEFAULT_VARIANT)
    method = choice.get("method")
    overrides = None

    if "path" in choice:
        try:
            overrides, follower = load_configuration_file(choice["path"])
        except (OSError, ValueError, yaml.YAMLError) as error:
            sg.popup_error(f"Could not read that configuration:\n\n{error}")
            return None, None

        if follower in ACCOMPANION_VARIANTS:
            variant_key = follower

        method = (overrides.get("score_follower_kwargs") or {}).get("score_follower")

    while True:
        variant = ACCOMPANION_VARIANTS[variant_key]
        acc_version = load_class(variant["module"], variant["class_name"])

        defaults = constructor_defaults(acc_version)

        follower_kwargs = score_follower_defaults(variant_key, method)

        if overrides:
            # A None in the file means 'not set', which the constructor's own
            # empty value already says in a type the GUI can render.
            defaults.update(
                {
                    k: v
                    for k, v in overrides.items()
                    if k in defaults and k != "score_follower_kwargs" and v is not None
                }
            )

            if isinstance(overrides.get("score_follower_kwargs"), dict):
                follower_kwargs.update(overrides["score_follower_kwargs"])

        defaults["score_follower_kwargs"] = follower_kwargs

        result = class_init_configurations_via_gui(
            acc_version,
            window_title=f"ACCompanion - {variant['class_name']}",
            subtitle=_subtitle(variant_key, follower_kwargs.get("score_follower")),
            hooks=accompanion_hooks(variant_key, pieces),
            defaults=defaults,
            extra_rows={"Piece": piece_rows(pieces)},
            save_tag=variant_key,
        )

        if result is None:
            print("Configuration aborted")
            return None, None

        if isinstance(result, ConfigurationReload):
            if result.follower in ACCOMPANION_VARIANTS:
                variant_key = result.follower

            overrides = result.config
            method = ((overrides or {}).get("score_follower_kwargs") or {}).get(
                "score_follower"
            )
            continue

        return result, acc_version


def _subtitle(variant_key, method):
    variant = ACCOMPANION_VARIANTS[variant_key]

    return f"{variant['class_name']}  -  score follower '{method}'"
