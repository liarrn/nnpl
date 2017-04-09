from nnpl.blob import *
from nnpl.layer_factory import *
from nnpl.layers import *

class Net:
    '''
    Param shoud be of format [{'name': 'layer1', 'bottoms': ['blob1', 'blob2'], 'tops': ['blob1', 'blob2'], 'param': param info}, ...]
    
    Blob names and layer names are used by Net to bookkeeping the blobs and layers to prevent use undefined blob or 
    double define blobs
    
    layers use param to initilize. use bottoms and tops to setup up and initilize for internal weights and bias
    '''
    
    def __init__(self, params):
        self.num_layers_ = len(params)
        self.layer_names_ = [layer['name'] for layer in params]
        self.layers_ = []
        
        self.blobs_ = []
        self.top_blobs_ = []
        self.bottom_blobs_ = []
        self.blob_name2idx_ = {}
        self.blob_names_ = []
        
        self.learnable_params_ = []
        
        for param in params:
            # initialize layer
            layer = layer_creator[param['type']](param)
            self.layers_.append(layer)
            bottoms = []
            tops = []
            
            # initialize bottom blobs
            for bottom_name in param['bottoms']:
                assert bottom_name in self.blob_names_, 'blob %s has not defined'%bottom_name
                blob_id = self.blob_name2idx_[bottom_name]
                bottoms.append(self.blobs_[blob_id])
            self.bottom_blobs_.append(bottoms)
            
            # initialize top blobs
            for top_name in param['tops']:
                if top_name in self.blob_names_:
                    if top_name in bottoms:
                        # inplace layer, to be implemented
                        raise NnplException('blob %s has already been defined. Not support inplace layer yet'%top_name)
                    if top_name not in bottoms:
                        raise NnplException('blob %s has already been defined.'%top_name)
                else:
                    # blob top_name has not been defined
                    blob_id = len(self.blob_names_)
                    self.blob_name2idx_[top_name] = blob_id
                    self.blob_names_.append(top_name)
                    top = Blob()
                    tops.append(top)
                    self.blobs_.append(top)
            self.top_blobs_.append(tops)
            
            # setup layer
            layer.setup(bottoms, tops)
            
            #set learnable_params
            if layer.blobs_ != None:
                for blob in layer.blobs_:
                    if blob.is_update_:
                        self.learnable_params_.append(blob)
        return
    
    def forward(self):
        loss = 0.0
        for i in range(self.num_layers_):
            layer = self.layers_[i]
            layer.forward(self.bottom_blobs_[i], self.top_blobs_[i])
#             loss += layer.forward(self.bottom_blobs_[i], self.top_blobs_[i])
        return
    
    def backward(self):
        for i in reversed(range(self.num_layers_)):
            layer = self.layers_[i]
            layer.backward(self.bottom_blobs_[i], self.top_blobs_[i])
        return
    
    def forward_backward(self):
        self.forward()
        self.backward()
        return
    
    def update(self):
        for layer in self.layers_:
            if layer.blobs_ == None:
                continue
            for blob in layer.blobs_:
                blob.update()