layer_creator = {}
def layer_register(layer_name, layer):
    assert layer_name not in layer_creator.keys(), 'layer %s has already been defined'%layer_name
    layer_creator[layer_name] = layer
	
def layer_unregister(layer_name):
    if layer_name in layer_creator:
        del layer_creator[layer_name]