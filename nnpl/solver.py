from nnpl.net import *

class Solver:
    def __init__(self, param):
        self.net_ = Net(param['net'])
        self.type_ = param['type']
        self.lr_rate_ = param['lr_rate']
        self.max_iter_ = param['max_iter']
        return
    
    def solve(self):
        self.step(self.max_iter_)
        return
    
    def step(self, iters):
        for i in range(iters):
            self.net_.forward_backward()
            for blob in self.net_.learnable_params_:
                blob.diff_ *= self.lr_rate_
            self.net_.update()
        return