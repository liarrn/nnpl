from nnpl.filler import *
from nnpl.layer_factory import *
import numpy as np
import copy 

layer_unregister('InnerProduct')

class InnerProductLayer:
    
    def __init__(self, param):
        self.output_size_ = param['output_size']
        self.weight_filler_param_ = copy.deepcopy(param['weight_filler_param'])
        self.bias_filler_param_ = copy.deepcopy(param['bias_filler_param'])
        self.input_size_ = None
        self.batch_size_ = None
#         self.weight_ = None
#         self.bias_ = None
        self.blobs_ = [None, None]  # [weight_, bias_]
        return
        
    def setup(self, bottoms, tops):
        assert len(bottoms) == 1, 'layer InnerProductLayer only support bottoms of length 1'
        assert len(tops) == 1, 'layer InnerProductLayer only support tops of length 1'
        data = bottoms[0].data_
        self.batch_size_, self.input_size_ = data.shape
        tops[0].setup((self.batch_size_, self.output_size_))
        weight_filler = Filler(self.weight_filler_param_)
        self.blobs_[0] = weight_filler.fill((self.input_size_, self.output_size_))
        bias_filler = Filler(self.bias_filler_param_)
        self.blobs_[1] = bias_filler.fill((self.output_size_, ))
        return
    
    def forward(self, bottoms, tops):
        tops[0].data_ = np.dot(bottoms[0].data_, self.blobs_[0].data_) + self.blobs_[1].data_
        return
        
    def backward(self, bottoms, tops):
        bottoms[0].diff_ = np.dot(self.blobs_[0].data_, tops[0].diff_.T).T
        self.blobs_[0].diff_ = np.dot(bottoms[0].data_.T, tops[0].diff_)
        self.blobs_[1].diff_ = np.sum(tops[0].diff_, axis=0)
        return

layer_register('InnerProduct', InnerProductLayer)