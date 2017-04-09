class DataTransform:
    '''TODO: lots of optimization '''
    
    def __init__(self, param):
        self.radial_param_ = copy.deepcopy(param['radial'])
        self.angle_param_ = copy.deepcopy(param['angle'])
        self.dist_mat = None
        return
    
    def infer_feature_size(self):
        radial_size = reduce(lambda count, item: count * len(item), self.radial_param_.values(), 1)
        angular_size = reduce(lambda count, item: count * len(item), self.angle_param_.values(), 1)
        return radial_size + angular_size
    
    def soft_cutoff(self, r, rc):
        ''' 
        a soft cutoff function with respect of inter-atom length r
        from Behler et. al. PRL 98, 146401 (2007)
        '''
        if r > rc:
            return 0
        if r <= rc:
            return 0.5 * (np.cos(np.pi * r / rc) + 1)
        
    def dist(self, a1, a2):
        '''distance between two atoms'''
        return np.sqrt(np.sum((a1-a2)**2))
    
    def angle (self, ca, ra1, ra2):
        '''angle between center atom and two reference atoms'''
        b1 = ra1 - ca
        b2 = ra2 - ca
        tmp = np.dot(b1, b2) / (self.dist(ca, ra1) * self.dist(ca, ra2))
        tmp = np.clip(tmp, -1, 1)
        return np.arccos(tmp)
    
    def transform(self, coor):
        return np.hstack((self.radial_symm_func(coor), self.angular_symm_func(coor)))
    
    def radial_symm_func(self, coor):
        '''Radial symmetry functions'''
        #         args_list = make_grid(eta_list, rs_list, rc_list)
        args_list = make_grid(self.radial_param_['eta_list'], self.radial_param_['rs_list'], self.radial_param_['rc_list'])
        num_atoms = coor.shape[0]
        result = np.zeros((num_atoms, len(args_list)))
        for center_atom in range(num_atoms):
            for args_num in range(len(args_list)):
                eta, rs, rc = args_list[args_num]
                for ref_atom in range(num_atoms):
                    # ra stands for reference atom
                    if ref_atom == center_atom:
                        continue
                    dist_ref_center = self.dist(coor[ref_atom], coor[center_atom])
                    # symmetry function between center atom and reference atom
                    gcr = self.soft_cutoff(dist_ref_center, rc) * np.exp(-eta*(dist_ref_center-rs)**2)
                    result[center_atom, args_num]  += gcr
        return result
    
    def angular_symm_func(self, coor):
        '''Angular symmetry functions'''
        #         args_list = make_grid(eta_list, lambda_list, zeta_list, rc_list)
        args_list = make_grid(self.angle_param_['eta_list'], self.angle_param_['lambda_list'], 
                              self.angle_param_['zeta_list'], self.angle_param_['rc_list']) 
        num_atoms = coor.shape[0]
        result = np.zeros((num_atoms, len(args_list)))
        for center_atom in range(num_atoms):
            for args_num in range(len(args_list)):
                eta, lambda_, zeta, rc = args_list[args_num]
                for ref_atom_1 in range(num_atoms):
                    for ref_atom_2 in range(num_atoms):
                        if ref_atom_1 == center_atom or ref_atom_2 == center_atom:
                            continue
                        dist_ref1_center = self.dist(coor[ref_atom_1], coor[center_atom])
                        dist_ref2_center = self.dist(coor[ref_atom_2], coor[center_atom])
                        dist_ref1_ref2 = self.dist(coor[ref_atom_1], coor[ref_atom_2])
                        # symmetry function 
                        if center_atom == 0:
                            self.angle(coor[center_atom], coor[ref_atom_1], coor[ref_atom_2])
#                             print np.cos(self.angle(coor[center_atom], coor[ref_atom_1], coor[ref_atom_2]))
                        gcr = self.soft_cutoff(dist_ref1_center, rc) * self.soft_cutoff(dist_ref2_center, rc) * self.soft_cutoff(dist_ref1_ref2, rc)
                        gcr *= 2 ** (1 - zeta)
                        gcr *= (1 + lambda_ * np.cos(self.angle(coor[center_atom], coor[ref_atom_1], coor[ref_atom_2]))) ** zeta
                        gcr *= np.exp(-eta*(dist_ref1_center**2+dist_ref2_center**2+dist_ref1_ref2**2))
                        result[center_atom, args_num]  += gcr
        return result


eta_list = [0.03, 0.09]
lambda_list = [-1.0, 1.0]
zeta_list = [1.0, 2.0, 4.0]
rc_list = [6.0]
# angular_symm_func(coor, eta_list, lambda_list, zeta_list, rc_list)

eta_list = [0.12, 0.2]
rs_list = [0.0]
rc_list = [6.0]
# radial_symm_func(coor, eta_list, rs_list, rc_list)

coor = np.zeros((3, 3))
coor[1] = np.array([1, 0, 0])
coor[2] = np.array([0, 1, 0])
# print coor
with open('./coor/100.xyz', 'r') as fp:
    coor = fp.readlines()
coor = coor[2: ]
coor = np.array([map(float, line.split()[1: 4]) for line in coor])
# coor
param = {}
param['radial'] = {'eta_list':  [0.12, 0.2], 'rs_list': [0.0], 'rc_list': [6.0]}
param['angle'] = {'eta_list':  [0.03, 0.09], 'lambda_list':  [-1.0, 1.0],
                  'zeta_list': [1.0, 2.0, 4.0], 'rc_list': [6.0]}
# param['angle'] = {'eta_list':  [1], 'lambda_list':  [1.0],
#                   'zeta_list': [1.0], 'rc_list': [6.0]}
dt = DataTransform(param)
start_time = time.time()
b = dt.transform(coor)
elapsed_time = time.time() - start_time
print elapsed_time
# print coor
# print b
# dt.infer_feature_size()
# dt.angular_symm_func(coor) == angular_symm_func(coor, eta_list, lambda_list, zeta_list, rc_list)
# dt.radial_symm_func(coor) == radial_symm_func(coor, eta_list, rs_list, rc_list)