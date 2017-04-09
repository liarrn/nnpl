class TxtReader:
    '''
    very inefficent
    '''
    def __init__(self, file_path):
        self.file_path_ = file_path
        self.multiple_label_ = None
        with open(self.file_path_, 'r') as fp:
            line = fp.readline()
        if len(line.split(',')) == 1:
            self.multiple_label_=False
        else:
            self.multiple_label_=True
        return
    
    def fetch_data(self, line_num):
        with open(self.file_path_, 'r') as fp:
            line = fp.readlines()[line_num]
        if self.multiple_label_:
            result = np.array(map(float, line.split(',')))
        else:
            result = np.array(float(line.strip()))
        return result
        
    def fetch_data(self, start_line, end_line):
        with open(self.file_path_, 'r') as fp:
            lines = fp.readlines()[start_line: end_line]
        if self.multiple_label_:
            result = np.array([map(float, line.split(',')) for line in lines])
        else:
            result = np.array([float(line.strip()) for line in lines])
        return result
        return np.array([map(float, line.split(',')) for line in lines])
        
    def peek(self):
        with open(self.file_path_, 'r') as fp:
            line = fp.readline()
        if self.multiple_label_:
            result = np.array(map(float, line.split(',')))
        else:
            result = np.array(float(line.strip()))
        return result
    
# a = TxtReader('./iris/labels.dat')
# print a.multiple_label_
# a.fetch_data(0, 2).shape
