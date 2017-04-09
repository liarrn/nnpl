import numpy as np
import copy
from nnpl.blob import Blob

class Filler:
    def __init__(self, param):
        #         self.param_ = copy.deepcopy(param)
        self.type_ = param['type']
        self.param_ = copy.deepcopy(param)
        return
    
    def fill(self, shape):
        #         shape = container.shape
        size =  reduce(lambda count, item: count * item, shape, 1)
        weight = Blob()
        if self.type_ == 'gaussian':
            std = self.param_.get('std', 0.01)
            weight.data_ = np.random.randn(size).reshape(shape) * std
        elif self.type_ == 'constant':
            value = self.param_.get('value', 0.0)
            weight.data_ = np.ones(size).reshape(shape) * value
        return weight
    
# param = {'type': 'gaussian', 'std': 0.1}
# filler = Filler(param)
# w = filler.fill(np.array([20, 30]))
# plt.hist(w.reshape((-1)), bins=20)


# param = {'type': 'constant'}
# filler = Filler(param)
# w = filler.fill(np.array([20]))
# w