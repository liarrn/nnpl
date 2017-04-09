from nnpl.txt_reader import  *
from nnpl.layer_factory import *
import numpy as np

layer_unregister('BehlerTxtData')

class BehlerTxtDataLayer:
    '''
    TODO: support none value in feature file, support shuffle
    Feature data file: csv [num_feature_entries * num_features]
    Label data file: csv [num_label_entries * num_labels]
    For each line in label data file: [number of feature entry corresponding to the same label, label]
    assert sum(label[:, 0]) == feature.shape[0]
    The actual batch_size is less than the set batch size
    Adopted from Behler et. al. PRL 98, 146401 (2007)
    '''
    def __init__(self, param):
        '''{'feature_file': 'path/to/feature/file', 'label_file': 'path/to/label/file', 'batch_size': batch_size, 'num_entries': num_entries}'''
        self.blobs_ = None
        self.batch_size_ = param['batch_size']
        self.feature_reader_ = TxtReader(param['feature_file'])
        self.label_reader_ = TxtReader(param['label_file'])
        self.label_map_ = None  # map from label to features
        self.feature_idx_ = 0 # the current feature index
        self.label_idx_ = 0 # the current label index
        self.num_feature_entries_ = None # number of features (atoms)
        self.num_label_entries_ = None # number of labels (clusters)
        self.num_features_ = None
        self.num_labels_ = None
        return
    
    def setup(self, bottoms, tops):
        '''
        tops[0] for feature blob, tops[1] for label blob
        '''
        assert len(tops) == 2, 'layer TxtDataLayer only support bottoms of length 2'
        self.num_features_ = self.feature_reader_.peek().shape[0]
        self.num_labels_ = self.label_reader_.peek().shape[0]
        tops[0].setup((0, self.num_features_))
        tops[1].setup((0, self.num_labels_))
        
        tops[0].is_update_ = False
        tops[1].is_update_ = False
        
        self.label_map_ = self.label_reader_.fetch_all()[:, 0].astype(int)
        for i in range(1, self.label_map_.shape[0]):
            self.label_map_[i] += self.label_map_[i-1]
        self.num_feature_entries_ = self.label_map_[-1]
        self.num_label_entries_ = len(self.label_map_)
        return
    
    def forward(self, bottoms, tops):
        # store data in text file is highly inefficent, would better to use database like lmdb or leveldb
        # take care of when idx_+batch_size_ larger than num_entries
        feature_loops = []
        label_loops = []
        num_loops = (self.feature_idx_ + self.batch_size_) / self.num_feature_entries_
        feature_batch_size = 0  # actual feature batch size
        label_batch_size = 0 # actual label batch size
        if num_loops == 0:
            next_label_idx = np.searchsorted(self.label_map_, self.feature_idx_ + self.batch_size_)
            if next_label_idx == 0:
                assert False, 'batch size is too small, should be larger than the largest atom number in cluster'
            else:
                next_feature_idx = self.label_map_[next_label_idx-1]
            feature_batch_size = next_feature_idx - self.feature_idx_
            label_batch_size = next_label_idx - self.label_idx_
            feature_loops.append((self.feature_idx_, next_feature_idx))
            label_loops.append((self.label_idx_, next_label_idx))
            self.label_idx_ = next_label_idx
            self.feature_idx_ = next_feature_idx
        else:
            feature_loops.append((self.feature_idx_, self.num_feature_entries_))
            label_loops.append((self.label_idx_, self.num_label_entries_))
            feature_batch_size = self.num_feature_entries_ - self.feature_idx_
            label_batch_size = self.num_label_entries_ - self.label_idx_
            for i in range(1, num_loops):
                feature_batch_size += self.num_feature_entries_
                label_batch_size += self.num_label_entries_
                feature_loops.append((0, self.num_feature_entries_))
                label_loops.append((0, self.num_label_entries_))
            res_features = (self.feature_idx_ + self.batch_size_) % self.num_feature_entries_
            next_label_idx = np.searchsorted(self.label_map_, res_features)
            if next_label_idx == 0:
                next_feature_idx = 0
            else:
                next_feature_idx = self.label_map_[next_label_idx-1]
                feature_batch_size += next_feature_idx
                label_batch_size += next_label_idx
                feature_loops.append((0, next_feature_idx))
                label_loops.append((0, next_label_idx))
            self.label_idx_ = next_label_idx
            self.feature_idx_ = next_feature_idx
        
        tops[0].setup((feature_batch_size, self.num_features_))
        tops[1].setup((label_batch_size, self.num_labels_))
        feature_batch_idx = 0
        label_batch_idx = 0
        for i in range(len(feature_loops)):
            tops[0].data_[feature_batch_idx: feature_batch_idx + feature_loops[i][1] - feature_loops[i][0]] = \
                self.feature_reader_.fetch_data(feature_loops[i][0], feature_loops[i][1])
            tops[1].data_[label_batch_idx: label_batch_idx + label_loops[i][1] - label_loops[i][0]] = \
                self.label_reader_.fetch_data(label_loops[i][0], label_loops[i][1])
            feature_batch_idx = feature_batch_idx + feature_loops[i][1] - feature_loops[i][0]
            label_batch_idx = label_batch_idx + label_loops[i][1] - label_loops[i][0]
        return
    
    def backward(self, bottoms, tops):
        return

layer_register('BehlerTxtData', BehlerTxtDataLayer)
# txt_data_param = {'batch_size': 10, 'feature_file': './test/feature.txt', 'label_file': './test/label.txt'}
# tdl = BehlerTxtDataLayer(txt_data_param)


# fb, lb = Blob(), Blob()
# tops = [fb, lb]
# bottoms = None

# tdl.setup(bottoms, tops)
# print tdl.label_map_
# for i in range(10):
#     tdl.forward(bottoms, tops)
#     print 'features:\n', tops[0].data_
#     print 'labels: \n', tops[1].data_
#     print '\n\n'