import numpy as np

class TxtReader:
    '''
    very inefficent
    '''
    def __init__(self, file_path):
        self.file_path_ = file_path
        return
    
    def fetch_data(self, line_num):
        with open(self.file_path_, 'r') as fp:
            line = fp.readlines()[line_num]
        result = np.array(map(float, line.split(',')))
        return result
        
    def fetch_data(self, start_line, end_line):
        with open(self.file_path_, 'r') as fp:
            lines = fp.readlines()[start_line: end_line]
        return np.array([map(float, line.split(',')) for line in lines])
        
    def fetch_all(self):
        with open(self.file_path_, 'r') as fp:
            lines = fp.readlines()
        return np.array([map(float, line.split(',')) for line in lines])
        
    def peek(self):
        with open(self.file_path_, 'r') as fp:
            line = fp.readline()
        return np.array(map(float, line.split(',')))
        return result
    
# a = TxtReader('./iris/labels.dat')
# print a.multiple_label_
# a.fetch_data(0, 2).shape
