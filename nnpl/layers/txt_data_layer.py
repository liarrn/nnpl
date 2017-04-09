from nnpl.txt_reader import  *
from nnpl.layer_factory import *

layer_unregister('TxtData')

class TxtDataLayer:
    '''
    TODO: support none value in feature file, support shuffle
    data layer for inputs from text format
    expecting two txt files each for features and labels
    one line for one entry
    features should be seperated by commas
    only support single label for each entry
    '''
    def __init__(self, param):
        '''{'feature_file': 'path/to/feature/file', 'label_file': 'path/to/label/file', 'batch_size': batch_size, 'num_entries': num_entries}'''
        self.blobs_ = None
        self.batch_size_ = param['batch_size']
        self.feature_reader_ = TxtReader(param['feature_file'])
        self.label_reader_ = TxtReader(param['label_file'])
        
        self.idx_ = 0 # the current file number
        self.num_entries_ = param['num_entries'] # the number of entries in the feature file
        return
    
    def setup(self, bottoms, tops):
        '''
        tops[0] for feature blob, tops[1] for label blob
        '''
        assert len(tops) == 2, 'layer TxtDataLayer only support bottoms of length 2'
        num_features = self.feature_reader_.peek().shape[0]
        num_labels = self.label_reader_.peek().shape[0]
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
#             print loop
#             print tops[0].data_[batch_idx: batch_idx + loop[1] - loop[0]].shape
#             print self.feature_reader_.fetch_data(loop[0], loop[1]).shape
            tops[0].data_[batch_idx: batch_idx + loop[1] - loop[0]] = self.feature_reader_.fetch_data(loop[0], loop[1])
            tops[1].data_[batch_idx: batch_idx + loop[1] - loop[0]] = self.label_reader_.fetch_data(loop[0], loop[1])
            batch_idx = batch_idx + loop[1] - loop[0]
        return
    
    def backward(self, bottoms, tops):
        return

layer_register('TxtData', TxtDataLayer)
# txt_data_param = {'batch_size': 50, 'feature_file': './iris/features.dat', 'label_file': './iris/labels.dat', 'num_entries': 150}
# tdl = TxtDataLayer(txt_data_param)

# fb, lb = Blob(), Blob()
# tops = [fb, lb]
# bottoms = None

# tdl.setup(bottoms, tops)
# tdl.forward(bottoms, tops)
# print tops[0].data_[:2]
# print tops[1].data_[:2]
# tdl.forward(bottoms, tops)
# print tops[0].data_[:2]
# print tops[1].data_[:2]