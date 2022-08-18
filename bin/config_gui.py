from inspect import signature
from ast import literal_eval

import PySimpleGUI as gui
from typing import Union







def load_class(module_name,class_name):
	module = __import__(module_name,fromlist=[class_name])

	return getattr(module,class_name)

def class_init_args(class_init):
	s = signature(class_init)

	return [p for p in s.parameters.values()][1:]


def currently_supported_types(t):
	return t in (int,float,bool,list,dict,str)

_currently_supported_versions = dict(
	HMM_based=('accompanion.hmm_accompanion','HMMACCompanion'),
	OLTW_based=('accompanion.oltw_accompanion','OLTWACCompanion')
)


class Hook(object):
	__slots__=('trigger','configuration','layout','update','evaluation')
	'''
	A Hook object is intended to make it possible for users to provide or override functionality in the configuration GUI system

	Attributes:
		configuration: object -> ConfigurationNode
			this function is intended for transforming python objects into ConfigurationNodes which aren't currently supported (see currently_supported_types)
			can also be used to override the default transformation of supported types

		layout: (ConfigurationNode, enclosing_scope: str) -> [[PySimpleGUI.Element]]
			PySimpleGUI works by defining a grid of Elements
			this function transforms a ConfigurationNode into a list of rows of PySimpleGUI.Elements,
				which then get inserted into the overall layout of the GUI wherever the trigger occured
			with this function the look and interaction of/with parameters can be customized

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
	'''

	def __init__(self,trigger,configuration=None,layout=None,update=None,evaluation=None):
		if trigger is None:
			raise ValueError("a Hook object needs a trigger")

		if configuration is None and layout is None and update is None and evaluation is None:
			raise ValueError("a Hook object needs either a configuration, layout, update or evaluation function")

		self.trigger=trigger
		self.configuration=configuration
		self.layout=layout
		self.update=update
		self.evaluation=evaluation


def _retrieve(full_name,data_type,data,hooks):
	for i,trigger in enumerate(hooks['triggers']):
		if trigger(full_name,data_type,data):
			return hooks['functions'][i]
	return None


#The following functions define additional functionality in order to make ACCompanion configuration more convenient and are used via the Hook system

#####################################################################
def midi_router_kwargs_trigger(name,data_type,data):
	return name=='midi_router_kwargs'

def _in_out_port_distribution():
	from mido import get_input_names, get_output_names

	in_ports = get_input_names()
	out_ports = get_output_names()

	return (in_ports,out_ports,out_ports,out_ports,in_ports)

def midi_router_kwargs_configuration(value):
	port_names = [p.name for p in class_init_args(load_class('accompanion.midi_handler.midi_routing','MidiRouter').__init__)]

	distribution = _in_out_port_distribution()

	children = [(pn,ConfigurationNode(str,data=ports[0] if len(ports)>0 else '')) for pn,ports in zip(port_names,distribution)]

	return ConfigurationNode(dict,children=children)


def midi_router_kwargs_layout(config_node,enclosing_scope):
	port_names = [p.name for p in class_init_args(load_class('accompanion.midi_handler.midi_routing','MidiRouter').__init__)]

	distribution = _in_out_port_distribution()

	max_length = max([len(pn) for pn in port_names])

	layout=[[gui.Text(pn,size=(max_length,1)),gui.Combo(ports,default_value=ports[0] if len(ports)>0 else '',key=enclosing_scope+'.'+pn)] for pn,ports in zip(port_names,distribution)]

	return layout
####################################################################################


######################################################################################
def tempo_model_trigger(name,data_type,data):
	return name=='tempo_model_kwargs.tempo_model'

def tempo_model_configuration(value):
	import accompanion.accompanist.tempo_models as tempo_models

	sync_models = [a for a in dir(tempo_models) if 'SyncModel' in a]

	assert len(sync_models)>0, "can't load SyncModels if there are none in accompanion.accompanist.tempo_models"

	return ConfigurationNode(type,data=sync_models[0])

def tempo_model_layout(config_node,enclosing_scope):
	import accompanion.accompanist.tempo_models as tempo_models

	sync_models = [a for a in dir(tempo_models) if 'SyncModel' in a]

	if len(sync_models)==0:
		return []

	name = enclosing_scope.split('.')[-1]

	layout=[[gui.Text(name,size=(len(name),1)),gui.Combo(sync_models,default_value=sync_models[0],key=enclosing_scope)]]

	return layout

def tempo_model_eval(config_string):
	tempo_models = __import__('accompanion.accompanist.tempo_models',fromlist=[config_string])

	return getattr(tempo_models,config_string)
######################################################################################



#######################################################################################
def single_file_name_trigger(name,data_type,data):
	return 'fn' in name.split('.')[-1] and data_type is str

def accompaniment_match_trigger(name,data_type,data):
	return name=='accompaniment_match'

def single_file_name_layout(config_node,enclosing_scope):
	return [[gui.InputText(key=enclosing_scope),gui.FileBrowse(target=enclosing_scope)]]
#############################################################################################





##########################################################################################
def multiple_file_name_trigger(name,data_type,data):
	return 'fn' in name.split('.')[-1] and data_type in (list,)

def multiple_file_name_layout(config_node,enclosing_scope):
	return [[gui.Multiline(autoscroll=True,key=enclosing_scope),gui.FilesBrowse(enable_events=True,key=enclosing_scope+'_browse',target=enclosing_scope+'_browse',files_delimiter='\n')]]

def multiple_file_name_eval(config_string):
	return config_string.split('\n')


def multiple_file_name_browse_trigger(name,data_type,data):
	return 'fn' in name.split('.')[-1] and name[-len('_browse'):]=='_browse'

def multiple_file_name_browse_update(window,event,values):
	file_names=values[event].split('\n')

	if len(file_names)>0:
		window[event[:-len('_browse')]].set_size((max([len(fn) for fn in file_names]),len(file_names)))
		window[event[:-len('_browse')]].update('\n'.join(file_names))
####################################################################################################










class ConfigurationNode(object):
	__slots__=['type','children','data']

	def __init__(self,node_type,children=[],data=None):
		self.type=node_type
		self.children=children
		self.data=data

	def value(self):
		if self.type is dict:
			return {child_name:child.value() for child_name,child in self.children}
		else:
			return self.data

	def search(self,search_name):
		dot_loc = search_name.find('.')

		outer_scope = search_name[:dot_loc] if dot_loc>=0 else search_name

		for child_name,child in self.children:
			if outer_scope==child_name:
				if dot_loc<0:
					return child
				if child.type is dict:
					return child.search(search_name[dot_loc+1:])
				else:
					return None

		return None

def check_for_type_error(config_node, enclosing_scope=''):
	if not config_node.type is dict:
		if len(config_node.children)>0:
			raise ValueError(f"Node error at {enclosing_scope[1:]}\nNode is not of type dict, but has children {config_node.children}")
		elif type(config_node.data)!=config_node.type:
			raise TypeError(f"Type error at {enclosing_scope[1:]}\nNode is of type {config_node.type},\nbut value {config_node.data}\nis of type {type(config_node.data)}")
	elif len(config_node.children)>0:
		if not config_node.data is None:
			raise TypeError(f"Type error at {enclosing_scope[1:]}\nNode has children {config_node.children}, but also data {config_node.data}")

		for child_name, child in config_node.children:
			check_for_type_error(child,enclosing_scope+'.'+child_name)
	elif not type(config_node.data) is dict:
		raise TypeError(f"Type error at {enclosing_scope[1:]}\nNode is of type dict,\nbut value {config_node.data}\nis of type {type(config_node.data)}")





def configuration_tree(underlying_dict,configuration_hooks=dict(triggers=[],functions=[]),enclosing_scope=''):
	children = []

	for k,v in underlying_dict.items():
		

		configure = _retrieve(enclosing_scope+k,type(v),v,configuration_hooks)

		if configure is None and not currently_supported_types(type(v)):
			print(f"the configuration GUI currently doesn't support parameters of type {type(v)} and therefore silently ignores {enclosing_scope+k}")
			continue

		if not configure is None:
			child=configure(v)
		elif type(v) is dict and len(v)>0:
			child=configuration_tree(v,configuration_hooks,enclosing_scope+k+'.')
		else:
			child = ConfigurationNode(type(v),data=v)

		children.append((k,child))

	return ConfigurationNode(dict,children=children)




def gui_layout(config_node,layout_hooks=dict(triggers=[],functions=[]),enclosing_scope=''):
	field_names = [f'{child_name} : {str(child.type)[len("<class x"):-2]}' for child_name,child in config_node.children]
	max_length = max([len(f) for f in field_names])

	layout=[]

	for (child_name,child),f in zip(config_node.children,field_names):
		layout_hook = _retrieve(enclosing_scope+child_name,child.type,child.data,layout_hooks)
		
		if not layout_hook is None:
			sub_layout=layout_hook(child,enclosing_scope+child_name)
			
			layout.append([gui.Text(f,size=(max_length,1))])
			layout.extend([[gui.Text(size=(max_length,1))]+row for row in sub_layout])
		elif child.type is dict and len(child.children)>0:
			layout.append([gui.Text(f,size=(max_length,1))])

			sub_layout = gui_layout(child,layout_hooks,enclosing_scope+child_name+'.')

			

			layout.extend([[gui.Text(size=(max_length,1))]+row for row in sub_layout])
		else:
			layout.append([gui.Text(f,size=(max_length,1)),gui.InputText(str(child.data) if not child.type is dict else '{}',key=enclosing_scope+child_name)])

	return layout

def default_instance(t):
	def get_origin(t):
		return getattr(t,'__origin__',None)
	def get_args(t):
		return getattr(t,'__args__',())

	if get_origin(t) is list:
		return []
	elif get_origin(t) is Union and type(None) in get_args(t):
		a,b = get_args(t)

		if not a is type(None):
			return a()
		elif not b is type(None):
			return b()
		else:
			raise TypeError('why on earth does there exist a parameter of Union[None,None]?!')
	else:
		return t()


def class_init_configurations_via_gui(
	class_object,
	window_title=None,
	hooks=[],
	type_checked=True,
):
	parameters = class_init_args(class_object.__init__)

	underlying_dict = {p.name:(p.default if not p.default in [p.empty,None] else (default_instance(p.annotation) if p.annotation!=p.empty else '')) for p in parameters}

	hook_init_args = class_init_args(Hook.__init__)

	hook_system = {p.name:dict(triggers=[],functions=[]) for p in hook_init_args}

	for hook in hooks:
		for p in hook_init_args:
			function = getattr(hook,p.name,None)

			if not function is None:
				hook_system[p.name]['triggers'].append(hook.trigger)
				hook_system[p.name]['functions'].append(function)





	config_tree = configuration_tree(underlying_dict,hook_system['configuration'])   

	main_layout = gui_layout(config_tree,hook_system['layout'])

	main_layout.append([gui.Button('OK')])

	if window_title is None:
		window_title = class_object.__name__+' configuration'
	
	main_window = gui.Window(window_title,main_layout)

	while True:
		event, values = main_window.read()

		update = _retrieve(event,str,values,hook_system['update'])

		if not update is None:
			update(main_window,event,values[event])
		elif event == gui.WINDOW_CLOSED:
			return None
		elif event == 'OK':
			for k in values.keys():
				result=config_tree.search(k)

				if result is None:
					continue

				if len(result.children)>0:
					raise ValueError(f"{k} was configured, but has children. That shouldn't be the case")

				evaluate = _retrieve(k,result.type,result.data,hook_system['evaluation'])

				if not evaluate is None:
					result.data = evaluate(values[k])
				elif result.type!=str:
					result.data=literal_eval(values[k])
				else:
					result.data=values[k]

			if type_checked:
				try:
					check_for_type_error(config_tree)
				except TypeError as e:
					error_layout = [[gui.Text(str(e))]]
					error_window = gui.Window('ERROR',error_layout)
					error_window.read()
					error_window.close()
					continue
			
			config=config_tree.value()
			main_window.close()
			return config
			



def accompanion_configurations_and_version_via_gui():
	version_layout = [[gui.Text('Please choose a version. Currently supported are:')]]

	for csf in _currently_supported_versions.keys():
		version_layout.append([gui.Button(csf)])

	init_window = gui.Window('ACCompanion version', version_layout)

	event, values = init_window.read()



	init_window.close()

	if event == gui.WINDOW_CLOSED:
		return None, None

	
	acc_version = load_class(_currently_supported_versions[event][0],_currently_supported_versions[event][1])
	


	configs = class_init_configurations_via_gui(
		acc_version,
		hooks=(
			Hook(midi_router_kwargs_trigger,layout=midi_router_kwargs_layout,configuration=midi_router_kwargs_configuration),
			Hook(tempo_model_trigger,layout=tempo_model_layout,configuration=tempo_model_configuration,evaluation=tempo_model_eval),
			Hook(single_file_name_trigger,layout=single_file_name_layout),
			Hook(multiple_file_name_trigger,layout=multiple_file_name_layout,evaluation=multiple_file_name_eval),
			Hook(accompaniment_match_trigger,layout=single_file_name_layout),
			Hook(multiple_file_name_browse_trigger,update=multiple_file_name_browse_update)
		)
	)

	return configs,acc_version