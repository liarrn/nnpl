from nnpl.layer_factory import *
import numpy as np

layer_unregister('EuclideanLoss')

class EuclideanLossLayer:
    '''
    bottoms[0] of shape (batch_size * num_values), predicted values
    bottoms[1] of shape(batch_size * num_values), ground truth values
    '''
    
    def __init__(self, param):
        self.blobs_ = None
        return
    
    def setup(self, bottoms, tops):
        assert len(bottoms) == 2, 'layer EuclideanLossLayer only support bottoms of length 2'
        return
    
    def forward(self, bottoms, tops):
        N = bottoms[0].data_.shape[0]
        loss = 0.5 * np.sum((bottoms[0].data_ - bottoms[1].data_) ** 2) / N
        print loss
        return
    
    def backward(self, bottoms, tops):
        N = bottoms[0].data_.shape[0]
        bottoms[0].diff_ = bottoms[0].data_ - bottoms[1].data_
        return



layer_register('EuclideanLoss', EuclideanLossLayer)

# bottoms = [None] * 2
# bottoms[0] = Blob()
# bottoms[0].data_ = np.random.randn(20, 40)
# bottoms[1] = Blob()
# bottoms[1].data_ = np.random.randn(20, 40)
# tops = []

# param = None
# el_layer = EuclideanLossLayer(param)
# el_layer.forward(bottoms, tops)
# el_layer.backward(bottoms, tops)