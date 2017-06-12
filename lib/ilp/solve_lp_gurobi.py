from gurobipy import *
import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import pdb
import sys

def unpack(data):
    dims, vals = data[0, 0]
    dims = dims[0]
    return dims, vals

def vec_to_dense(dims, vals):
    sparse_mat = ss.coo_matrix((vals[:, 1], (vals[:, 0], np.zeros(vals.shape[0]))), shape=(dims[0], 1))
    mat = np.array(sparse_mat.todense())
    return mat[:, 0]

def load(filename=None):
    if filename is None:
        filename = 'A_b_obj.mat'
    data = sio.loadmat(filename)
    A_dims, A_vals = unpack(data["A"])
    b_dims, b_vals = unpack(data["b"])
    intflag_dims, intflag_vals = unpack(data["int_flag"])
    obj_dims, obj_vals = unpack(data["objective"])

    b = vec_to_dense(b_dims, b_vals)
    obj = vec_to_dense(obj_dims, obj_vals)
    try:
        intflag = intflag_vals[:,0]
    except:
        intflag = []

    A = ss.csr_matrix((A_vals[:, 2], (A_vals[:, 0], A_vals[:, 1])), shape=A_dims)

    return A_dims, A, b, obj, intflag

def solve_gurobi(filename=None):

    m = make_task(filename)
    
    # integer_delta = 100
    # n_reps = 0
    # while integer_delta > 0.01:
    #     print "reps: ", n_reps
    #     n_reps += 1
    #     task.putclist(range(num_vars), obj + 100*np.random.rand(num_vars))
    vprint('optimizing')
    m.params.MIPFocus = 1
    m.params.PreSolve = 2
    m.params.Symmetry = 2
    m.params.Disconnected = 2
    m.params.TimeLimit = 4*60*60
    m.params.BranchDir = -1
    m.params.GURO_PAR_Degenmoves = 0
    err_code = m.optimize()
    if m.Status == 9:
        fval = np.inf
        xx = np.zeros(m.numVars)
    else:
        try:    
            fval = m.ObjVal
            xx = [v.x for v in m.getVars()]
        except:
            fval = np.inf
            xx = np.zeros(m.numVars)

    
    return xx, fval
    

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def vprint(text):
    verbose = True
    if verbose:
        print text

def make_task(filename):
    from mat2mps import to_mps
    
    output_mps = '%s.mps' % os.path.splitext(filename)[0]
    
    with open(output_mps,'wb') as f:
        to_mps(filename, f)
    
    #command = 'python mat2mps.py %s > %s' % (filename, output_mps)
    #os.system(command)
    m = read(output_mps)
    
    return m
    
    