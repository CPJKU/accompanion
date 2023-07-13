from inspect import signature
from ast import literal_eval
import PySimpleGUI
from typing import Union
import os


def load_class(module_name, class_name):
    module = __import__(module_name, fromlist=[class_name])

    return getattr(module, class_name)


def class_init_args(class_init):
    s = signature(class_init)

    return [p for p in s.parameters.values()][1:]


def currently_supported_types(t):
    return t in (int, float, bool, list, dict, str)


_currently_supported_versions = {
    "HMM based (for solo simple Pieces)": ('accompanion.hmm_accompanion', 'HMMACCompanion'),
    "OLTW based (usually for four hands pieces)": ('accompanion.oltw_accompanion', 'OLTWACCompanion')
}


class ConfigurationNode(object):
    __slots__ = ['type', 'child_names_and_children', 'data']

    '''
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
    '''

    def __init__(self, node_type, child_names_and_children=[], data=None):
        self.type = node_type
        self.child_names_and_children = child_names_and_children
        self.data = data

    def value(self):
        if self.type is dict and len(self.child_names_and_children) > 0:
            return {child_name: child.value() for child_name, child in self.child_names_and_children}
        else:
            return self.data

    def search(self, search_name):
        dot_loc = search_name.find('.')

        outer_scope = search_name[:dot_loc] if dot_loc >= 0 else search_name

        for child_name, child in self.child_names_and_children:
            if outer_scope == child_name:
                if dot_loc < 0:
                    return child
                if child.type is dict:
                    return child.search(search_name[dot_loc + 1:])
                else:
                    return None

        return None


def check_for_type_error(config_node, enclosing_scope=''):
    if not config_node.type is dict:
        if len(config_node.child_names_and_children) > 0:
            raise TypeError(
                f"Node error at {enclosing_scope[1:]}\nNode is not of type dict, but has children {config_node.child_names_and_children}")
        elif type(config_node.data) != config_node.type:
            raise TypeError(
                f"Type error at {enclosing_scope[1:]}\nNode is of type {config_node.type},\nbut value {config_node.data}\nis of type {type(config_node.data)}")
    elif len(config_node.child_names_and_children) > 0:
        if not config_node.data is None:
            raise TypeError(
                f"Type error at {enclosing_scope[1:]}\nNode has children {config_node.child_names_and_children}, but also data {config_node.data}")

        for child_name, child in config_node.child_names_and_children:
            check_for_type_error(child, enclosing_scope + '.' + child_name)
    elif not type(config_node.data) is dict:
        raise TypeError(
            f"Type error at {enclosing_scope[1:]}\nNode is of type dict,\nbut value {config_node.data}\nis of type {type(config_node.data)}")


class Hook(object):
    __slots__ = ('trigger', 'configuration', 'layout', 'update', 'evaluation')
    '''
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

        update: (PySimpleGUI.Window, event: str, values: dict) -> void
            this function is somewhat of a workaround of PySimpleGUIs limited capabilities
            window.read is used in the configuration GUI and that method returns an event (in the form of a string)
                and	either a list or a dict of values (usually a dict, for details see PySimpleGUI docs)
            if a PySimpleGUI.Button is part of your layout, assigning that Button a key attribute (for details see PySimpleGUI docs),
                results in the event string being the key if the Button is pressed and in the values being a dict which can be looked up with the event
                furthermore, a Button can be assigned the keys of targets which are used as storage for the Button's result,
                which again can be looked up with the target key in the values dict
            using the event and values dict, one can update the GUI window accordingly (for details on how to update a PySimpleGUI.Window, see the docs)
            this function is intended to adapt the window layout during configuration while the layout function is for defining the initial layout
            this separation makes sense since PySimpleGUI offers limited ways to change a window once it has been initialized

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
    '''

    def __init__(self, trigger, configuration=None, layout=None, update=None, evaluation=None):
        if trigger is None:
            raise ValueError("a Hook object needs a trigger")

        if configuration is None and layout is None and update is None and evaluation is None:
            raise ValueError("a Hook object needs either a configuration, layout, update or evaluation function")

        self.trigger = trigger
        self.configuration = configuration
        self.layout = layout
        self.update = update
        self.evaluation = evaluation


def _retrieve(full_name, data_type, data, hooks):
    for i, trigger in enumerate(hooks['triggers']):
        if trigger(full_name, data_type, data):
            return hooks['functions'][i]
    return None


def configuration_tree(underlying_dict, configuration_hooks=dict(triggers=[], functions=[]), enclosing_scope=''):
    child_names_and_children = []

    for k, v in underlying_dict.items():

        configure = _retrieve(enclosing_scope + k, type(v), v, configuration_hooks)

        if configure is None and not currently_supported_types(type(v)):
            print(
                f"the configuration GUI currently doesn't support parameters of type {type(v)} and therefore silently ignores {enclosing_scope + k}")
            continue

        if not configure is None:
            child = configure(v)
        elif type(v) is dict and len(v) > 0:
            child = configuration_tree(v, configuration_hooks, enclosing_scope + k + '.')
        else:
            child = ConfigurationNode(type(v), data=v)

        child_names_and_children.append((k, child))

    return ConfigurationNode(dict, child_names_and_children=child_names_and_children)


def Collapsable(layout, key):
    return PySimpleGUI.pin(PySimpleGUI.Column(layout, key=key + 'collapsable'))


def gui_layout(config_node, layout_hooks=dict(triggers=[], functions=[]), enclosing_scope=''):
    field_names = [f'{child_name} : {str(child.type)[len("<class ") + 1:-2]}' for child_name, child in
                   config_node.child_names_and_children]
    max_length = max([len(f) for f in field_names])

    layout = []

    for (child_name, child), f in zip(config_node.child_names_and_children, field_names):
        def integrate_sub_layout(sub_layout):
            layout.append([PySimpleGUI.pin(PySimpleGUI.Button(f, key=enclosing_scope + child_name + 'toggle',
                                                              target=enclosing_scope + child_name + 'toggle'))])
            sub_layout = [[PySimpleGUI.Text(size=(max_length, 1))] + row for row in sub_layout]

            layout.append([Collapsable(sub_layout, enclosing_scope + child_name)])

        layout_hook = _retrieve(enclosing_scope + child_name, child.type, child.data, layout_hooks)

        if not layout_hook is None:
            sub_layout = layout_hook(child, enclosing_scope + child_name)

            integrate_sub_layout(sub_layout)
        elif child.type is dict and len(child.child_names_and_children) > 0:
            sub_layout = gui_layout(child, layout_hooks, enclosing_scope + child_name + '.')

            integrate_sub_layout(sub_layout)
        else:
            layout.append([Collapsable([[PySimpleGUI.Text(f, size=(max_length, 1)),
                                         PySimpleGUI.InputText(str(child.data) if not child.type is dict else '{}',
                                                               key=enclosing_scope + child_name)]],
                                       enclosing_scope + child_name)])

    return layout


# The following functions define additional functionality in order to make ACCompanion configuration more convenient and are used via the Hook system

#####################################################################
def midi_router_kwargs_trigger(name, data_type, data):
    return name == 'midi_router_kwargs'


def _in_out_port_distribution():
    from mido import get_input_names, get_output_names

    in_ports = get_input_names()
    out_ports = get_output_names()

    return (in_ports, out_ports, out_ports, out_ports, in_ports)


def midi_router_kwargs_configuration(value):
    assert type(value) is dict, "midi_router_kwargs_configuration was expected to be a dict"

    port_names = [p.name for p in
                  class_init_args(load_class('accompanion.midi_handler.midi_routing', 'MidiRouter').__init__)]

    distribution = _in_out_port_distribution()

    child_names_and_children = []

    for port_name, ports in zip(port_names, distribution):
        data = ''
        if port_name in value.keys():
            data = value[port_name]

            assert len(data) == 0 or data in ports, f"{data} is not a port that is found among viable ports {ports}"
        elif len(ports) > 0:
            data = ports[0]

        child_names_and_children.append((port_name, ConfigurationNode(str, data=data)))

    return ConfigurationNode(dict, child_names_and_children=child_names_and_children)


def midi_router_kwargs_layout(config_node, enclosing_scope):
    # port_names = [p.name for p in class_init_args(load_class('accompanion.midi_handler.midi_routing','MidiRouter').__init__)]

    # distribution = _in_out_port_distribution()

    # max_length = max([len(pn) for pn in port_names])

    # layout=[[gui.Text(pn,size=(max_length,1)),gui.Combo(ports,default_value=ports[0] if len(ports)>0 else '',key=enclosing_scope+'.'+pn)] for pn,ports in zip(port_names,distribution)]

    max_length = max([len(port_name) for port_name, _ in config_node.child_names_and_children])

    layout = []

    for (port_name, child), ports in zip(config_node.child_names_and_children, _in_out_port_distribution()):
        if len(child.data) > 0:
            try:
                pos = ports.index(child.data)
                ports[pos], ports[0] = ports[0], ports[pos]
            except ValueError:
                raise ValueError(f"{child.data} is not a port that is found among viable ports {ports}")

        combo_list = ports

        layout.append([PySimpleGUI.Text(port_name, size=(max_length, 1), key=enclosing_scope + '.' + port_name + 'name'),
                       PySimpleGUI.Combo(combo_list, default_value=combo_list[0] if len(child.data) > 0 else '',
                                         key=enclosing_scope + '.' + port_name)])

    return layout


####################################################################################


######################################################################################
def tempo_model_trigger(name, data_type, data):
    return name == 'tempo_model_kwargs.tempo_model'


def tempo_model_configuration(value):
    import accompanion.accompanist.tempo_models as tempo_models

    sync_model_names = [a for a in dir(tempo_models) if 'SyncModel' in a]

    assert len(sync_model_names) > 0, "can't load SyncModels if there are none in accompanion.accompanist.tempo_models"

    if type(value) is type:
        data = value.__name__
    else:
        data = value

    assert data in sync_model_names or len(
        data) == 0, f"default value {value} is neither an empty string nor something that can be found among the SyncModels"

    return ConfigurationNode(type, data=data)


def tempo_model_layout(config_node, enclosing_scope):
    import accompanion.accompanist.tempo_models as tempo_models

    sync_model_names = [a for a in dir(tempo_models) if 'SyncModel' in a]

    input_name = config_node.data if type(config_node.data) is str else config_node.data.__name__

    assert len(
        input_name) == 0 or input_name in sync_model_names, f"at config_node {enclosing_scope} default value {config_node.data} is neither an empty string nor something that can be found among the SyncModels"

    if len(sync_model_names) == 0:
        return []

    if len(input_name) > 0:
        pos = sync_model_names.index(input_name)

        sync_model_names[0], sync_model_names[pos] = sync_model_names[pos], sync_model_names[0]

    name = enclosing_scope.split('.')[-1]

    layout = [
        [
            PySimpleGUI.Text(name, size=(len(name), 1), key=enclosing_scope + 'name'),
            PySimpleGUI.Combo(sync_model_names, default_value=sync_model_names[0] if len(input_name) > 0 else '',
                              key=enclosing_scope)
        ]
    ]

    return layout


def tempo_model_eval(config_string):
    tempo_models = __import__('accompanion.accompanist.tempo_models', fromlist=[config_string])

    return getattr(tempo_models, config_string)


######################################################################################


# TODO: set initial_folder to sample_pieces once a folder structure is established


#######################################################################################
def single_file_name_trigger(name, data_type, data):
    return 'fn' in name.split('.')[-1] and data_type is str


def accompaniment_match_trigger(name, data_type, data):
    return name == 'accompaniment_match'


def single_file_name_layout(config_node, enclosing_scope):
    return [[PySimpleGUI.InputText(config_node.data, key=enclosing_scope),
             PySimpleGUI.FileBrowse(target=enclosing_scope, key=enclosing_scope + '_browse')]]


#############################################################################################


##########################################################################################
def multiple_file_name_trigger(name, data_type, data):
    return 'fn' in name.split('.')[-1] and data_type in (list,)


def multiple_file_name_layout(config_node, enclosing_scope):
    width = max([len(x) for x in config_node.data]) if len(config_node.data) > 0 else None
    height = len(config_node.data) if len(config_node.data) > 0 else None
    return [[PySimpleGUI.Multiline('\n'.join(config_node.data), autoscroll=True, key=enclosing_scope, enable_events=True,
                                   size=(width, height)),
             PySimpleGUI.FilesBrowse(enable_events=True, key=enclosing_scope + '_browse', target=enclosing_scope + '_browse',
                                     files_delimiter='\n')]]


def multiple_file_name_eval(config_string):
    return config_string.split('\n') if len(config_string) > 0 else []


def multiple_file_name_browse_trigger(name, data_type, data):
    return 'fn' in name.split('.')[-1] and name[-len('_browse'):] == '_browse'


def multiple_file_name_browse_update(window, event, values):
    file_names = values[event].split('\n')

    if len(file_names) > 0:
        window[event[:-len('_browse')]].set_size((max([len(fn) for fn in file_names]), len(file_names)))
        window[event[:-len('_browse')]].update('\n'.join(file_names))


####################################################################################################


def default_instance(t):
    def get_origin(t):
        return getattr(t, '__origin__', None)

    def get_args(t):
        return getattr(t, '__args__', ())

    if get_origin(t) is list:
        return []
    elif get_origin(t) is Union and type(None) in get_args(t):
        a, b = get_args(t)

        if not a is type(None):
            return a()
        elif not b is type(None):
            return b()
        else:
            raise TypeError('why on earth does there exist a parameter of Union[None,None]?!')
    else:
        return t()


def _create_config(values, config_tree, evaluation_hooks, type_checked):
    for k in values.keys():
        result = config_tree.search(k)

        if result is None:
            continue

        if len(result.child_names_and_children) > 0:
            raise ValueError(f"{k} was configured, but has children. That shouldn't be the case")

        evaluate = _retrieve(k, result.type, result.data, evaluation_hooks)

        if not evaluate is None:
            result.data = evaluate(values[k])
        elif result.type != str:
            result.data = literal_eval(values[k])
        else:
            result.data = values[k]

    if type_checked:
        try:
            check_for_type_error(config_tree)
        except TypeError as e:
            error_layout = [[PySimpleGUI.Text(str(e))]]
            error_window = PySimpleGUI.Window('ERROR', error_layout)
            error_window.read()
            error_window.close()
            return None

    return config_tree.value()


def class_init_configurations_via_gui(
        class_object,
        window_title=None,
        hooks=[],
        type_checked=True,
):
    parameters = class_init_args(class_object.__init__)

    underlying_dict = {p.name: (p.default if not p.default in [p.empty, None] else (
        default_instance(p.annotation) if p.annotation != p.empty else '')) for p in parameters}

    hook_init_args = [p.name for p in class_init_args(Hook.__init__) if p.name != 'trigger']

    hook_system = {name: dict(triggers=[], functions=[]) for name in hook_init_args}

    for hook in hooks:
        for name in hook_init_args:
            function = getattr(hook, name, None)

            if not function is None:
                hook_system[name]['triggers'].append(hook.trigger)
                hook_system[name]['functions'].append(function)

    if window_title is None:
        window_title = class_object.__name__ + ' configuration'

    main_window = PySimpleGUI.Window('')

    while True:
        config_tree = configuration_tree(underlying_dict, hook_system['configuration'])

        main_layout = gui_layout(config_tree, hook_system['layout'])

        if 'gui_config_files' in os.listdir(os.getcwd()):
            gui_config_file_dir = "./gui_config_files"
        else:
            gui_config_file_dir = "../gui_config_files"

        header = PySimpleGUI.Frame('Menu', [[PySimpleGUI.Button('Configuration finished', key='config done'),
                                             PySimpleGUI.Button('Save Configuration to File', key='save config'),
                                             PySimpleGUI.FileBrowse('Load Configuration from File', key='load config',
                                                                    target='load config', enable_events=True,
                                                                    initial_folder=gui_config_file_dir)]])

        main_layout = [[header]] + main_layout

        dispose_window = main_window
        main_window = PySimpleGUI.Window(window_title, main_layout)

        # print('finalize window')
        main_window.finalize()
        # print('dispose of window')
        dispose_window.close()
        # print('done')

        while True:
            event, values = main_window.read()

            if event == PySimpleGUI.WINDOW_CLOSED:
                print("Configuration aborted")
                main_window.close()
                return None

            update = _retrieve(event, str, values, hook_system['update'])

            if not update is None:
                update(main_window, event, values)
            elif event[-len('toggle'):] == 'toggle':
                target = event[:-len('toggle')]

                main_window[event].metadata = object() if main_window[event].metadata is None else None

                for element in main_window.key_dict.keys():
                    if element[:len(target)] == target and element[-len('collapsable'):] == 'collapsable':
                        # print(element)

                        main_window[element].update(visible=(main_window[event].metadata is None))

            # print('-----------------------------------------------------')
            elif event == 'load config':
                with open(values[event], 'r') as config_file:
                    import yaml

                    underlying_dict = yaml.unsafe_load(config_file)

                    break
            elif event == 'save config':
                config = _create_config(values, config_tree, hook_system['evaluation'], type_checked)

                if config is None:
                    continue

                file_id = len(os.listdir(gui_config_file_dir))

                with open(f"{gui_config_file_dir}/{class_object.__name__}_configuration{file_id}.yaml", 'w') as dest:
                    import yaml
                    yaml.dump(config, dest)
            elif event == 'config done':
                config = _create_config(values, config_tree, hook_system['evaluation'], type_checked)

                if config is None:
                    continue

                main_window.close()
                return config


def accompanion_configurations_and_version_via_gui():
    version_layout = [[PySimpleGUI.Text('Please choose a version. Currently supported are:')]]

    for csf in _currently_supported_versions.keys():
        version_layout.append([PySimpleGUI.Button(csf)])

    init_window = PySimpleGUI.Window('ACCompanion version', version_layout)

    event, values = init_window.read()

    init_window.close()

    if event == PySimpleGUI.WINDOW_CLOSED:
        print("Configuration aborted")
        return None, None

    acc_version = load_class(_currently_supported_versions[event][0], _currently_supported_versions[event][1])

    configs = class_init_configurations_via_gui(
        acc_version,
        hooks=(
            Hook(midi_router_kwargs_trigger, layout=midi_router_kwargs_layout,
                 configuration=midi_router_kwargs_configuration),
            Hook(tempo_model_trigger, layout=tempo_model_layout, configuration=tempo_model_configuration,
                 evaluation=tempo_model_eval),
            Hook(single_file_name_trigger, layout=single_file_name_layout),
            Hook(multiple_file_name_trigger, layout=multiple_file_name_layout, evaluation=multiple_file_name_eval),
            Hook(accompaniment_match_trigger, layout=single_file_name_layout),
            Hook(multiple_file_name_browse_trigger, update=multiple_file_name_browse_update)
        )
    )

    return configs, acc_version