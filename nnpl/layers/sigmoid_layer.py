from nnpl.layer_factory import *
import numpy as np

layer_unregister('Sigmoid')

class SigmoidLayer:
    def __init__(self, param):
        self.blobs_ = None
        return
    
    def setup(self, bottoms, tops):
        assert len(bottoms) == 1, 'layer SigmoidLayer only support bottoms of length 1'
        assert len(tops) == 1, 'layer SigmoidLayer only support tops of length 1'
        tops[0].setup(bottoms[0].shape())
        return
    
    def forward(self, bottoms, tops):
        tops[0].data_ = 1.0 / (1.0 + np.exp(-bottoms[0].data_))
        return
    
    def backward(self, bottoms, tops):
        sig_diff = (1.0 - tops[0].data_) * tops[0].data_
        bottoms[0].diff_ = tops[0].diff_ * sig_diff
        return
layer_register('Sigmoid', SigmoidLayer)
    
# bottoms = [None]
# bottoms[0] = Blob()
# bottoms[0].data_ = np.random.randn(20, 40)
# tops = [None]
# tops[0] = Blob()
# tops[0].diff_ = np.ones((20, 40))

# sig_param = {'name': 'sig'}
# sig_layer = SigmoidLayer(sig_param)
# sig_layer.forward(bottoms, tops)
# plt.plot(bottoms[0].data_.reshape(-1), tops[0].data_.reshape(-1), 'x')
# sig_layer.backward(bottoms, tops)
# plt.plot(bottoms[0].data_.reshape(-1), bottoms[0].diff_.reshape(-1), 'x')