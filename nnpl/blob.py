import numpy as np

class Blob:
    def __init__(self):
        self.data_ = None
        self.diff_ = None
        self.is_update_ = True
        return
    
    def setup(self, shape):
        num_args = len(shape)
        assert num_args == 1 or num_args == 2, 'the number of arguments should either be 1 or 2'
        self.data_ = np.empty(shape, dtype=float)  # of size (natoms#1+natoms#2+...) * feature_size
        self.diff_ = np.empty(shape, dtype=float) # of same size of self.coors
        return
    
    def shape(self):
        assert self.data_.shape == self.diff_.shape, 'the shape of data_ does not match with the shape of diff_'
        return self.data_.shape
    
    def update(self):
        if self.is_update_:
#             print 'updating'
#             print 'before: ', self.data_[0, 0]
            assert self.data_.shape == self.diff_.shape, 'the shape of data_ does not match with the shape of diff_'
            self.data_ -= self.diff_
#             print 'after: ', self.data_[0, 0]
        return