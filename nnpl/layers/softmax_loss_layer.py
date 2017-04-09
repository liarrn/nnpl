from nnpl.layer_factory import *
import numpy as np

layer_unregister('SoftmaxLoss')

class SoftmaxLossLayer:
    '''
    bottoms[0] of shape (batch_size * num_labels), unnormalized exp prob
    bottoms[1] of shape(batch_size * 1), ground truth label for each batch
    entries bottom[1] should be in range(0, num_labels)
    '''
    
    def __init__(self, param):
        self.prob_ = None
        self.blobs_ = None
        return
    
    def setup(self, bottoms, tops):
        assert len(bottoms) == 2, 'layer SoftmaxLossLayer only support bottoms of length 2'
        return
    
    def forward(self, bottoms, tops):
        self.prob_ = np.exp(bottoms[0].data_ - np.max(bottoms[0].data_, axis=1, keepdims=True))
        self.prob_ /= np.sum(self.prob_, axis=1, keepdims=True)
        N = bottoms[0].data_.shape[0]
        loss = -np.sum(np.log(self.prob_[np.arange(N), list(bottoms[1].data_.reshape(-1).astype(int))])) / N
        print loss
        return
    
    def backward(self, bottoms, tops):
        bottoms[0].diff_ = self.prob_.copy()
        N = bottoms[0].data_.shape[0]
        bottoms[0].diff_[np.arange(N), list(bottoms[1].data_.reshape(-1).astype(int))] -= 1
        bottoms[0].diff_ /= N
        return

layer_register('SoftmaxLoss', SoftmaxLossLayer)

# bottoms = [None] * 2
# bottoms[0] = Blob()
# bottoms[0].data_ = np.random.randn(20, 40)
# bottoms[1] = Blob()
# bottoms[1].data_ = np.random.randint(40, size=20)
# tops = []

# sm_layer = SoftmaxLossLayer()
# sm_layer.forward(bottoms, tops)
# sm_layer.backward(bottoms, tops)
# plt.plot(bottoms[0].data_[0, :], sm_layer.prob_[0, :], 'x')