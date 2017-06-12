import mosek
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

def solve_mosek(filename=None):
    A_dims, A, b, obj, intflag = load(filename)
    num_constraints, num_vars = A_dims
    print "Loaded", A_dims
    
    integer_solver = True
    
    task = make_task(A, b, obj, intflag, integer_solver)

    vprint('optimizing')
    task.optimize()
    task.solutionsummary(mosek.streamtype.msg)
    xx = np.zeros(num_vars, float)
    soltype = mosek.soltype.itg if integer_solver else mosek.soltype.itr
    task.getxx(soltype, xx)
    fval = task.getprimalobj(soltype)
   
    return xx, fval, obj#, task
    

def solve_mosek_nearby(task, nearby_partial_soln):   
    num_vars = task.getnumvar()
    
    n_new_vars = 2*len(nearby_partial_soln)
    
    task.appendvars(n_new_vars)   
    task.putboundlist(mosek.accmode.var, range(num_vars,num_vars+n_new_vars), 
        [mosek.boundkey.lo]*n_new_vars, [0]*n_new_vars, [np.inf]*n_new_vars)

    num_cons = task.getnumcon()
    n_new_cons = len(nearby_partial_soln)
    task.appendcons(n_new_cons)
    (nearby_var_idx, nearby_val) = zip(*nearby_partial_soln)
    task.putboundlist(mosek.accmode.con, range(num_cons,num_cons+n_new_cons), 
        [mosek.boundkey.fx]*n_new_cons, nearby_val, nearby_val) 
    
    for i,row_idx in enumerate(range(num_cons,num_cons+n_new_cons)):
        task.putarow(row_idx, [nearby_var_idx[i], num_vars+2*i,num_vars+2*i+1], [1,-1,1]) 
    
    task.putclist(range(num_vars), np.zeros(n_new_vars))
    task.putclist(range(num_vars,num_vars+n_new_vars), np.ones(n_new_vars))
    
    vprint('optimizing')
    task.optimize()
    task.solutionsummary(mosek.streamtype.msg)
    xx = np.zeros(num_vars+n_new_vars, float)
    soltype = mosek.soltype.itr
    task.getxx(soltype, xx)
    fval = task.getprimalobj(soltype)
        
    task.removevars(range(num_vars,num_vars+n_new_vars))
    task.removecons(range(num_cons,num_cons+n_new_cons))
    
    return xx, fval, task

# Define a stream printer to grab output from MOSEK
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def vprint(text):
    verbose = True
    if verbose:
        print text

def make_task(A, b, obj, intflag, integer_solver):
    
    vprint('starting mosek environment...')
    env = mosek.Env()
    env.set_Stream(mosek.streamtype.log, streamprinter)
    
    vprint('creating mosek task...')
    task = env.Task(0,0)
    task.set_Stream(mosek.streamtype.log, streamprinter)

    num_constraints, num_vars = A.shape
    
    # set problem size
    task.appendcons(num_constraints)
    task.appendvars(num_vars)

    vprint('creating objective...')
    task.putclist(range(num_vars), obj)

    vprint('creating bounds...')
    for j in xrange(num_vars):
        # 0 <= x_j <= np.inf
        #task.putbound(mosek.accmode.var, j, mosek.boundkey.lo, 0, np.inf)
        task.putbound(mosek.accmode.var, j, mosek.boundkey.ra, 0, 1)

    vprint('creating equalities...')
    boundkeys = [mosek.boundkey.fx]*num_constraints  
    #pdb.set_trace()  
    task.putboundlist(mosek.accmode.con, range(num_constraints), boundkeys, b, b)
    [i_s, j_s, v_s] = ss.find(A)
    task.putaijlist(i_s, j_s, v_s)
    # we want to maximize
    task.putobjsense(mosek.objsense.maximize)
    
    if integer_solver:
        # define the integer variables
        task.putvartypelist(intflag, [mosek.variabletype.type_int]*len(intflag))
    
    # turn off basis identification
    task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
    return task
    
    