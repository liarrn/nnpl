'''
Read xyz data from feature_file_path
Use BehlerDataTransform transform xyz data
Output features for each atom in feature_file_path
Output number of atoms and energy for each cluster in label_file_path
'''
import sys
import time
import numpy as np
sys.path.append('../../')
from nnpl.transformers.behler_data_transform import BehlerDataTransform

# source_file_path = './2-subset.xyz'
# feature_file_path = './2-features.dat'
# label_file_path = './2-labels.dat'
source_file_path = './demo.xyz'
feature_file_path = './demo-features.dat'
label_file_path = './demo-labels.dat'
epa = -5758.103436562  # energy per atom

param = {}
param['radial'] = {'eta_list':  [0.003, 0.03, 0.06, 0.12, 0.2, 0.35, 0.7, 1.4], 'rs_list': [0.0], 'rc_list': [6.5]}
param['angle'] = {'eta_list':  [0.0003, 0.03, 0.09], 'lambda_list':  [-1.0, 1.0],
                  'zeta_list': [1.0, 2.0, 4.0], 'rc_list': [6.5]}
dt = BehlerDataTransform(param)

start_time = time.time()
with open(source_file_path) as fp:
    lines = fp.readlines()
print 'time when finished reading xyz data', time.time() - start_time
ln = 0
nc = 0
with open(feature_file_path, 'w') as fp: 
    pass
with open(label_file_path, 'w') as lp:
    pass
with open(feature_file_path, 'a') as fp:
        with open(label_file_path, 'a') as lp:
            while ln < len(lines):
                print 'processing the %d cluster'%nc
                nc += 1
                natom = int(lines[ln])
                energy = float(lines[ln+1].split()[-2])
                energy -= epa * natom
                label = np.array([[natom, energy]])
                coor = np.array([map(float, line.split()[1: 4]) for line in lines[ln+2: ln + natom + 2]])
                feature = dt.transform(coor)
                ln = ln + natom + 2
                np.savetxt(fp, feature, delimiter=',', fmt='%.8f', newline='\n')
                np.savetxt(lp, label, delimiter=',', fmt='%.8f', newline='\n')
print 'time when finished transforming and writing', time.time() - start_time

# start_time = time.time()
# with open(source_file_path) as fp:
#     lines = fp.readlines()
# print 'time when finished reading xyz data', time.time() - start_time
# ln = 0
# accum_natoms = 0  # accumulated natoms
# labels = np.empty((0, 2), dtype=float)
# features = np.empty((0, dt.infer_feature_size()), dtype=float)
# while ln < len(lines):
#     natom = int(lines[ln])
#     energy = float(lines[ln+1].split()[-2])
#     label = np.array([[natom, energy]])
#     coor = np.array([map(float, line.split()[1: 4]) for line in lines[ln+2: ln + natom + 2]])
#     feature = dt.transform(coor)
#     features = np.concatenate((features, feature), axis=0)
#     labels = np.concatenate((labels, label), axis=0)
#     ln = ln + natom + 2
    
#     accum_natoms += natom
#     if accum_natoms >= 10000:
#         accum_natoms = 0
#         with open(feature_file_path, 'a') as fp:
#             with open(label_file_path, 'a') as lp:
#                 np.savetxt(fp, features, delimiter=',', fmt='%.8f', newline='\n')
#                 np.savetxt(lp, labels, delimiter=',', fmt='%.8f', newline='\n')
    
# print 'time when finished transforming and writing', time.time() - start_time