from nnpl.layer_factory import *
import numpy as np

layer_unregister('BehlerEuclideanLoss')

class BehlerEuclideanLossLayer:
    '''
    Adopted from Behler et. al. PRL 98, 146401 (2007)
    bottoms[0] of shape (batch_size * 1), predicted output values
    bottoms[1] of shape [num_label_entries * 2], ground truth values
    For each line in bottoms[1]: [number of ouput entry corresponding to the same label, label]
    assert sum(label[:, 0]) == output.shape[0]
    '''
    
    def __init__(self, param):
        self.blobs_ = None
        output2label_mat = None
        return
    
    def setup(self, bottoms, tops):
        assert len(bottoms) == 2, 'layer BehlerEuclideanLossLayer only support bottoms of length 2'
        assert bottoms[0].data_.shape[1] == 1, 'layer BehlerEuclideanLossLayer only support one output for each input'
        assert bottoms[1].data_.shape[1] == 2, 'layer BehlerEuclideanLossLayer only support [num_label_entries * 2]'
        return
    
    def forward(self, bottoms, tops):
        num_output_to_label = bottoms[1].data_[:, 0].astype(int)
        num_labels = bottoms[1].data_.shape[0]
        num_outputs = bottoms[0].data_.shape[0]
        assert np.sum(num_output_to_label) == num_outputs, 'Error'
        ouput2label_map = np.zeros(num_labels+1, dtype=int)
        for i in range(1, num_labels+1):
            ouput2label_map[i] = ouput2label_map[i-1] + num_output_to_label[i-1]
        output2label_mat = np.zeros((num_labels, num_outputs))
        print num_outputs
        for i in range(num_labels):
            output2label_mat[i, ouput2label_map[i]: ouput2label_map[i+1]] = 1
        pred_value = np.dot(output2label_mat, bottoms[0].data_)
        gt_value =  bottoms[1].data_[:, 1]
        loss = 0.5 * np.sum((pred_value - gt_value) ** 2) / num_outputs
        print loss
        return
    
    def backward(self, bottoms, tops):
        num_output_to_label = bottoms[1].data_[:, 0].astype(int)
        num_labels = bottoms[1].data_.shape[0]
        num_outputs = bottoms[0].data_.shape[0]
        assert np.sum(num_output_to_label) == num_outputs, 'Error'
        ouput2label_map = np.zeros(num_labels+1, dtype=int)
        for i in range(1, num_labels+1):
            ouput2label_map[i] = ouput2label_map[i-1] + num_output_to_label[i-1]
        output2label_mat = np.zeros((num_labels, num_outputs))
        for i in range(num_labels):
            output2label_mat[i, ouput2label_map[i]: ouput2label_map[i+1]] = 1
        pred_value = np.dot(output2label_mat, bottoms[0].data_)
        gt_value =  bottoms[1].data_[:, 1]
        tmp_diff = pred_value - gt_value
        bottoms[0].diff_ = np.dot(output2label_mat.T, tmp_diff)
        return



layer_register('BehlerEuclideanLoss', BehlerEuclideanLossLayer)

# bottoms = [None] * 2
# bottoms[0] = Blob()
# bottoms[0].data_ = np.random.randn(20, 1)
# bottoms[1] = Blob()
# bottoms[1].data_ = np.array([[15, 0], [5, 0]])
# tops = []

# param = None
# el_layer = BehlerEuclideanLossLayer(param)
# el_layer.forward(bottoms, tops)
# el_layer.backward(bottoms, tops)