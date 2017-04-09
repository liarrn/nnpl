from nnpl.layer_factory import *
import copy

layer_unregister('MemData')


class MemDataLayer:
    '''
    top[0] for feature data
    top[1] for label data
    '''
    def __init__(self, param):
        self.blobs_ = None
        self.batch_size_ = param['batch_size']
        self.features_ = copy.deepcopy(param['features'])
        self.labels_ = copy.deepcopy(param['labels'])
        assert self.features_.shape[0] == self.labels_.shape[0], 'the size of feature data and label data does not match'
        
        self.idx_ = 0 # the current file number
        self.num_entries_ = self.features_.shape[0]  # the number of entries in the feature file
        return
    
    def setup(self, bottoms, tops):
        '''
        tops[0] for feature blob, tops[1] for label blob
        '''
        assert len(tops) == 2, 'layer TxtDataLayer only support bottoms of length 2'
        num_features = self.features_.shape[1]
        num_labels = self.labels_.shape[1]
        tops[0].setup((self.batch_size_, num_features))
        tops[1].setup((self.batch_size_, num_labels))
        tops[0].is_update_ = False
        tops[1].is_update_ = False
        return
    
    def forward(self, bottoms, tops):
        # store data in text file is highly inefficent, would better to use database like lmdb or leveldb
        # take care of when idx_+batch_size_ larger than num_entries
        loops = []
        num_loops = (self.idx_ + self.batch_size_) / self.num_entries_
        if num_loops == 0:
            loops.append((self.idx_, self.idx_+self.batch_size_))
        else:
            loops.append((self.idx_, self.num_entries_))
            for i in range(1, num_loops):
                loops.append((0, self.num_entries_))
            if (self.idx_ + self.batch_size_) % self.num_entries_ != 0:
                loops.append((0, (self.idx_ + self.batch_size_) % self.num_entries_))
        self.idx_ = (self.idx_ + self.batch_size_) % self.num_entries_
        
        batch_idx = 0
        for loop in loops:
            tops[0].data_[batch_idx: batch_idx + loop[1] - loop[0], :] = self.features_[loop[0]: loop[1], :]
            tops[1].data_[batch_idx: batch_idx + loop[1] - loop[0], :] = self.labels_[loop[0]: loop[1], :]
            batch_idx = batch_idx + loop[1] - loop[0]
        return
    
    def backward(self, bottoms, tops):
        return

layer_register('MemData', MemDataLayer)

# mem_data_param = {}
# mem_data_param['batch_size'] = 4
# mem_data_param['features'] = np.random.rand(10, 2)
# mem_data_param['labels'] = np.random.rand(10)

# bottoms = []
# features = Blob()
# labels = Blob()
# tops = [features, labels]

# m = MemDataLayer(mem_data_param)
# m.setup(bottoms, tops)

# print m.features_
# m.forward(bottoms, tops)
# print tops[0].data_
# m.forward(bottoms, tops)
# print tops[0].data_