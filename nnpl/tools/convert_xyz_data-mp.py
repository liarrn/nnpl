'''
Read xyz data from feature_file_path
Use BehlerDataTransform transform xyz data
Output features for each atom in feature_file_path
Output number of atoms and energy for each cluster in label_file_path
'''
import sys
import time
import numpy as np
import os
from multiprocessing import Process
sys.path.append('../../')
from nnpl.transformers.behler_data_transform import BehlerDataTransform

def split_xyzfile(in_filename, nproc, outdir='./'):
    with open(in_filename) as fp:
        lines = fp.readlines()
    nline = len(lines)
    sline = [0] * nproc  # start lines
    lstep = nline / nproc  # line step
    for i in range(1, nproc):
        eline = sline[i-1] + lstep
        while (len(lines[eline].split()) != 1):
            eline += 1
        sline[i] = eline
    for i in range(nproc):
        with open(os.path.join(outdir, '%02d-proc.xyz')%i, 'w') as fp:
            if i != (nproc-1):
                fp.write(''.join(lines[sline[i]: sline[i+1]]))
            else:
                fp.write(''.join(lines[sline[i]: ]))
    return

def dt_worker(param, proc_num, outdir = './'):
    source_file_path = os.path.join(outdir, '%02d-proc.xyz'%proc_num)
    feature_file_path = os.path.join(outdir, '%02d-features.dat'%proc_num)
    label_file_path = os.path.join(outdir, '%02d-labels.dat'%proc_num)
    dt = BehlerDataTransform(param)
    
    start_time = time.time()
    with open(source_file_path) as fp:
        lines = fp.readlines()
    print 'proc %d: time when finished reading xyz data %.3f s\n'%(proc_num, time.time() - start_time)
    ln = 0
    nc = 0
    with open(feature_file_path, 'a') as fp:
        with open(label_file_path, 'a') as lp:
            while ln < len(lines):
                if nc % 1000 == 0:
                    print 'proc %d: processing the %d cluster\n'%(proc_num, nc)
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
    print 'proc %d: time when finished transforming and writing %.3f s\n'%(proc_num, time.time() - start_time)
    return
    
def reduce_features(nproc, outdir='./'):
    with open(os.path.join(outdir, 'features.dat'), 'w') as fpo:
        with open(os.path.join(outdir, 'labels.dat'), 'w') as lpo:
            for i in range(nproc):
                with open(os.path.join(outdir, '%02d-features.dat'%i), 'r') as fpi:
                    fpo.write(fpi.read())
                with open(os.path.join(outdir, '%02d-labels.dat'%i), 'r') as lpi:
                    lpo.write(lpi.read())
    
    
if __name__ == '__main__':
    nproc = 24
    xyz_file = './2-subset.xyz'
    outdir = './data/'
    epa = -5758.103436562  # energy per atom

    param = {}
    param['radial'] = {'eta_list':  [0.003, 0.03, 0.06, 0.12, 0.2, 0.35, 0.7, 1.4], 'rs_list': [0.0], 'rc_list': [6.5]}
    param['angle'] = {'eta_list':  [0.0003, 0.03, 0.09], 'lambda_list':  [-1.0, 1.0],
                    'zeta_list': [1.0, 2.0, 4.0], 'rc_list': [6.5]}
    
    split_xyzfile(xyz_file, nproc, outdir)
    
    proc_list = [None] * nproc
    for i in range(nproc):
        p = Process(target=dt_worker, args=(param, i, outdir))
        proc_list[i] = p
        p.start()
    for i in range(nproc):
        proc_list[i].join()
    
    reduce_features(nproc, outdir)
    
    