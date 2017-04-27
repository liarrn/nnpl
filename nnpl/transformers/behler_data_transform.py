import copy
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

# make grid of list of arguments
# suppose we have 4 arguments, each has [M, N, P, Q] entries
# the q arguments vary the fastest
# for an arbitrary permute [m, n, p, q], the index is 
# q + pQ + nPQ + mNPQ = q+Q(p+P(n+mN))
def make_grid(*args):
    num_var = len(args)
    num_permut = reduce(lambda count, arg: count * len(arg), args, 1)
    result = np.zeros((num_permut, num_var))
    for i in range(num_permut):
        index = i
        for j in reversed(range(num_var)):
            sub_index = index % len(args[j])
            result[i, j] = args[j][sub_index]
            index /= len(args[j])
    return result

# a =  range(0, 2)
# b = range(2, 4)
# c = range(4, 6)
# make_grid(a, b, c)

class BehlerDataTransform:
    '''
    TODO: lots of optimization 
    TODO: only support single rc cutoff for both angular symm func and radial symm func
    Adopted from Behler et. al. PRL 98, 146401 (2007)
    
    '''
    
    def __init__(self, param):
        assert len(param['radial']['rc_list']) == 1 and len(param['angle']['rc_list']) ==1 and \
            param['angle']['rc_list'] == param['radial']['rc_list'], \
            'only support single rc'
        self.radial_param_ = copy.deepcopy(param['radial'])
        self.angle_param_ = copy.deepcopy(param['angle'])
        self.dist_mat_ = None
        self.soft_cutoff_mat_ = None
        self.cos_mat_ = None
        self.radial_symm_func_ = None
        self.angular_symm_func_ = None
        return
    
    def setup(self, coor):
        self.calc_dist_mat(coor)
        self.calc_soft_cutoff_mat(coor)
        self.calc_cos_mat(coor)
        return
    
    def reset(self):
        self.dist_mat_ = None
        self.soft_cutoff_mat_ = None
        self.cos_mat_ = None
        self.radial_symm_func_ = None
        self.angular_symm_func_ = None
        return
    
    def infer_feature_size(self):
        radial_size = reduce(lambda count, item: count * len(item), self.radial_param_.values(), 1)
        angular_size = reduce(lambda count, item: count * len(item), self.angle_param_.values(), 1)
        return radial_size + angular_size
    
    def calc_soft_cutoff_mat(self, coor):
        ''' 
        calculate soft cutoff matrix with respect of inter-atom length r
        '''
        assert isinstance(self.dist_mat_, np.ndarray), 'dis_mat_ should be calculated first'
        rc = self.radial_param_['rc_list'][0]
        self.soft_cutoff_mat_ = np.zeros_like(self.dist_mat_)
        self.soft_cutoff_mat_[self.dist_mat_ <= rc] = 0.5 * (np.cos(np.pi * self.dist_mat_[self.dist_mat_ <= rc] / rc) + 1)
        return
    
    def calc_dist_mat(self, coor):
        '''
        calculate distance matrix
        '''
        d = pdist(coor, metric='euclidean')
        self.dist_mat_ = squareform(d)
        return
    
    def calc_cos_mat (self, coor):
        '''
        calculate the cosine matrix 
        '''
        assert isinstance(self.dist_mat_, np.ndarray), 'dis_mat_ should be calculated first'
        N = coor.shape[0]
        self.cos_mat_ = np.empty((N, N, N))
        for i in range(N):
            numerator = np.dot(coor - coor[i, :], (coor - coor[i, :]).T)
            denominator = np.dot(self.dist_mat_[i, :].reshape((N, 1)), self.dist_mat_[i, :].reshape((1, N)))
            denominator[i, :] = 1 # mute divide zero error
            denominator[:, i] = 1 #  mute divide zero error
            self.cos_mat_[i] = numerator / denominator
            self.cos_mat_[i][i, :] = 0
            self.cos_mat_[i][:, i] = 0
        return
    
    def calc_radial_symm_func(self, coor):
        '''
        Radial symmetry functions
        '''
        assert isinstance(self.dist_mat_, np.ndarray), 'dis_mat_ should be calculated first'
        assert isinstance(self.soft_cutoff_mat_, np.ndarray), 'soft_cutoff_mat_ should be calculated first'
        args_list = make_grid(self.radial_param_['eta_list'], self.radial_param_['rs_list'], self.radial_param_['rc_list'])
        N = coor.shape[0]
        self.radial_symm_func_ = np.empty((N, len(args_list)))
        for args_num in range(len(args_list)):
            eta, rs, rc = args_list[args_num]
            dist_mat = self.dist_mat_.copy()
            soft_cutoff_mat = self.soft_cutoff_mat_.copy()
            np.fill_diagonal(dist_mat, rs)
            np.fill_diagonal(soft_cutoff_mat, 0)
            self.radial_symm_func_[:, args_num] = np.sum(np.exp(-eta * (dist_mat - rs) ** 2) * soft_cutoff_mat, axis=1)
        return
    
    def calc_angular_symm_func(self, coor):
        '''
        Angular symmetry functions
        '''
        assert isinstance(self.dist_mat_, np.ndarray), 'dis_mat_ should be calculated first'
        assert isinstance(self.soft_cutoff_mat_, np.ndarray), 'soft_cutoff_mat_ should be calculated first'
        assert isinstance(self.cos_mat_, np.ndarray), 'cos_mat_ should be calculated first'
        args_list = make_grid(self.angle_param_['eta_list'], self.angle_param_['lambda_list'], 
                              self.angle_param_['zeta_list'], self.angle_param_['rc_list']) 
        N = coor.shape[0]
        self.angular_symm_func_ = np.empty((N, len(args_list)))
        square_dist_mat = self.dist_mat_ ** 2
        
        for i in range(N):
            fc = self.soft_cutoff_mat_ * self.soft_cutoff_mat_[i, :].reshape((1, N)) * self.soft_cutoff_mat_[i, :].reshape((N, 1))
            fc[i, :] = 0
            fc[:, i] = 0
            dsq = square_dist_mat + square_dist_mat[i, :].reshape((1, N)) + square_dist_mat[i, :].reshape((N, 1))
            dsq[i, :] = 0
            dsq[:, i] = 0
            for args_num in range(len(args_list)):
                eta, lambda_, zeta, rc = args_list[args_num]
                self.angular_symm_func_[i, args_num] = 2 ** (1-zeta) * np.sum((1+lambda_*self.cos_mat_[i])**zeta * np.exp(-eta*dsq) * fc )
        return
    
    def transform(self, coor):
        self.setup(coor)
        self.calc_radial_symm_func(coor)
        self.calc_angular_symm_func(coor)
        result = np.hstack((self.radial_symm_func_, self.angular_symm_func_))
        self.reset()
        return result


# with open('./coor/100.xyz', 'r') as fp:
    # coor = fp.readlines()
# coor = coor[2: ]
# coor = np.array([map(float, line.split()[1: 4]) for line in coor])
# param = {}
# param['radial'] = {'eta_list':  [0.12, 0.2], 'rs_list': [0.0], 'rc_list': [6.0]}
# param['angle'] = {'eta_list':  [0.03, 0.09], 'lambda_list':  [-1.0, 1.0],
                  # 'zeta_list': [1.0, 2.0, 4.0], 'rc_list': [6.0]}
# dt = BehlerDataTransform(param)
# start_time = time.time()
# a = dt.transform(coor)
# elapsed_time = time.time() - start_time
# print elapsed_time
# dt.infer_feature_size()