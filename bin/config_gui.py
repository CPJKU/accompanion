import inspect
import ast

import PySimpleGUI as gui
import typing

def load_class(module_name,clas_name):
	module = __import__(module_name,fromlist=[clas_name])

	return getattr(module,clas_name)

def class_init_args(class_init):
	s = inspect.signature(class_init)

	return [p for p in s.parameters.values()][1:]




currently_supported_versions = dict(
	HMM_based=('accompanion.hmm_accompanion','HMMACCompanion'),
	OLTW_based=('accompanion.oltw_accompanion','OLTWACCompanion')
)





#####################################################################
def midi_ports_trigger(name,data_type,data):
	return name=='midi_router_kwargs'

def midi_ports_configuration(value):
	from mido import get_input_names, get_output_names

	in_ports = get_input_names()
	out_ports = get_output_names()

	port_names = [p.name for p in class_init_args(load_class('accompanion.midi_handler.midi_routing','MidiRouter').__init__)]

	distribution = [in_ports,out_ports,out_ports,in_ports,in_ports]

	children = [(pn,ConfigurationNode(str,data=ports[0] if len(ports)>0 else '')) for pn,ports in zip(port_names,distribution)]

	return ConfigurationNode(dict,children=children)


def midi_ports_layout(config_node,enclosing_scope):
	from mido import get_input_names, get_output_names

	in_ports = get_input_names()
	out_ports = get_output_names()

	port_names = [p.name for p in class_init_args(load_class('accompanion.midi_handler.midi_routing','MidiRouter').__init__)]

	distribution = [in_ports,out_ports,out_ports,in_ports,in_ports]

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
	return 'fn' in name.split('.')[-1] and data_type in (list,'event')

def multiple_file_name_layout(config_node,enclosing_scope):
	return [[gui.Multiline(autoscroll=True,key=enclosing_scope),gui.FilesBrowse(enable_events=True,key=enclosing_scope+'_browse',target=enclosing_scope+'_browse',files_delimiter='\n')]]

def multiple_file_name_update(window,event,value):
	file_names=value.split('\n')

	if len(file_names)>0:
		window[event[:-len('_browse')]].set_size((max([len(fn) for fn in file_names]),len(file_names)))
		window[event[:-len('_browse')]].update('\n'.join(file_names))

def multiple_file_name_eval(config_string):
	return config_string.split('\n')
####################################################################################################








configuration_hooks = {
	'triggers':[
		midi_ports_trigger,
		tempo_model_trigger
	],
	'functions':[
		midi_ports_configuration,
		tempo_model_configuration
	]
}

layout_hooks = {
	'triggers':[
		midi_ports_trigger,
		tempo_model_trigger,
		single_file_name_trigger,
		multiple_file_name_trigger,
		accompaniment_match_trigger
	],
	'functions':[
		midi_ports_layout,
		tempo_model_layout,
		single_file_name_layout,
		multiple_file_name_layout,
		single_file_name_layout
	]
}

update_hooks = {
	'triggers':[
		multiple_file_name_trigger
	],
	'functions':[
		multiple_file_name_update
	]
}

evaluation_hooks = {
	'triggers':[
		tempo_model_trigger,
		multiple_file_name_trigger
	],
	'functions':[
		tempo_model_eval,
		multiple_file_name_eval
	]
}




def currently_supported(t):
	return t in (int,float,bool,list,dict,str)



class ConfigurationNode(object):
	__slots__=['type','children','data']

	def __init__(self,node_type,children=[],data=None):
		self.type=node_type
		self.children=children
		self.data=data

	def check_for_type_error(self, enclosing_scope=''):
		if not self.type is dict:
			if len(self.children)>0:
				raise ValueError(f"Node error at {enclosing_scope[1:]}\nNode is not of type dict, but has children {self.children}")
			elif type(self.data)!=self.type:
				raise TypeError(f"Type error at {enclosing_scope[1:]}\nNode is of type {self.type},\nbut value {self.data}\nis of type {type(self.data)}")
		elif len(self.children)>0:
			if not self.data is None:
				raise TypeError(f"Type error at {enclosing_scope[1:]}\nNode has children {self.children}, but also data {self.data}")

			for child_name, child in self.children:
				child.check_for_type_error(enclosing_scope+'.'+child_name)
		elif not type(self.data) is dict:
			raise TypeError(f"Type error at {enclosing_scope[1:]}\nNode is of type dict,\nbut value {self.data}\nis of type {type(self.data)}")

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


def retrieve(full_name,data_type,data,hooks):
	for i,trigger in enumerate(hooks['triggers']):
		if trigger(full_name,data_type,data):
			return hooks['functions'][i]
	return None


def configuration_tree(underlying_dict,enclosing_scope=''):
	children = []

	for k,v in underlying_dict.items():
		

		configure = retrieve(enclosing_scope+k,type(v),v,configuration_hooks)

		if configure is None and not currently_supported(type(v)):
			print(f"the configuration GUI currently doesn't support parameters of type {type(v)} and therefore silently ignores {enclosing_scope+k}")
			continue

		if not configure is None:
			child=configure(v)
		elif type(v) is dict and len(v)>0:
			child=configuration_tree(v,enclosing_scope+k+'.')
		else:
			child = ConfigurationNode(type(v),data=v)

		children.append((k,child))

	return ConfigurationNode(dict,children=children)




def gui_layout(config_node,enclosing_scope=''):
	field_names = [f'{child_name} : {str(child.type)[len("<class x"):-2]}' for child_name,child in config_node.children]
	max_length = max([len(f) for f in field_names])

	layout=[]

	for (child_name,child),f in zip(config_node.children,field_names):
		layout_hook = retrieve(enclosing_scope+child_name,child.type,child.data,layout_hooks)
		
		if not layout_hook is None:
			sub_layout=layout_hook(child,enclosing_scope+child_name)
			
			layout.append([gui.Text(f,size=(max_length,1))])
			layout.extend([[gui.Text(size=(max_length,1))]+row for row in sub_layout])
		elif child.type is dict and len(child.children)>0:
			layout.append([gui.Text(f,size=(max_length,1))])

			sub_layout = gui_layout(child,enclosing_scope+child_name+'.')

			

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
	elif get_origin(t) is typing.Union and type(None) in get_args(t):
		a,b = get_args(t)

		if not a is type(None):
			return a()
		elif not b is type(None):
			return b()
		else:
			raise TypeError('why on earth does there exist a parameter of Union[None,None]?!')
	else:
		return t()




type_checked=True


def get_accompanion_arguments():
	window_title = 'ACCompanion configuration'

	

	version_layout = [[gui.Text('Please choose a version. Currently supported are:')]]

	for csf in currently_supported_versions.keys():
		version_layout.append([gui.Button(csf)])

	init_window = gui.Window(window_title, version_layout)

	event, values = init_window.read()



	init_window.close()

	
	acc_version = load_class(currently_supported_versions[event][0],currently_supported_versions[event][1])
	

	parameters = class_init_args(acc_version.__init__)

	config_tree = configuration_tree({p.name:(p.default if not p.default in [p.empty,None] else (default_instance(p.annotation) if p.annotation!=p.empty else '')) for p in parameters})   

	main_layout = gui_layout(config_tree)

	main_layout.append([gui.Button('OK')])
	
	main_window = gui.Window(window_title,main_layout)

	while True:
		event, values = main_window.read()

		#print(event, values)

		# #TODO: Introduce an EventType?
		update = retrieve(event,'event',values[event],update_hooks) if event in values.keys() else None

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

				evaluate = retrieve(k,result.type,result.data,evaluation_hooks)

				if not evaluate is None:
					result.data = evaluate(values[k])
				elif result.type!=str:
					try:
						result.data=ast.literal_eval(values[k])
					except SyntaxError as e:
						# print(k)
						# print(result.type)
						# print(values[k])
						raise e
				else:
					result.data=values[k]

			try:
				if type_checked:
					config_tree.check_for_type_error()
				config=config_tree.value()
				main_window.close()
				return config, acc_version
			except TypeError as e:
				error_layout = [[gui.Text(str(e))]]
				error_window = gui.Window('ERROR',error_layout)
				error_window.read()
				error_window.close()
