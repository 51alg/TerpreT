import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib", 'ilp'))
import numpy as np
import scipy.io as sio
import pdb
import pickle as pkl
import json
from numpy_encoder import NumpyEncoder


def solve_matlab(file_name):#Aeq, beq, objective, lb, ub):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    # add path to runLP.m
    eng.addpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib", 'ilp'))
    # add path to model
    eng.addpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.dirname(file_name)))
    x_and_fval = eng.runLP(os.path.basename(file_name), nargout=2)
    return x_and_fval
    
# def solve_backslash(file_name):
#     import matlab.engine
#     eng = matlab.engine.start_matlab()
#     #note that runLP.m must be in the same directory as this file
#     eng.addpath(os.path.dirname(os.path.realpath(__file__)))
#     x_and_fval = eng.solveBackslash(file_name, nargout=2)
#     return x_and_fval
        

def select_from_solution(var, x):
    return np.reshape(x[var['start']:(var['start']+var['length'])], var['dims'])

def export_json(file_name, dump_dict):
    json_out = open(file_name,'w')
    json_out.write(json.dumps(dump_dict, cls=NumpyEncoder,sort_keys=True, indent=4, separators=(',', ': ')))
    json_out.close()
    print ' saved solution to %s'%file_name
    
def find_contributions_to_objective(obj, x, var_dict):
    for i in range(len(obj)):
        contribution = obj[i]*x[i] 
        if contribution < -1:
            contributing_var = [var for _, var in var_dict.iteritems() if ((var['start'] <= i) and (var['start']+var['length'] > i))]
            pdb.set_trace()
            print contribution, contributing_var[0]['name'] 
            

def solve(in_file_name, out_file_name=None, solver='gurobi'):
    if solver == 'matlab':
        (x,fval) = solve_matlab(in_file_name)
        #(x,fval) = solve_backslash(in_file_name)
        x = np.array(x)
    elif solver == 'or':
        from solve_lp_ortools import solve_or
        (x, fval) = solve_or(in_file_name)
        x = np.array(x)
    elif solver == 'mosek':
        from solve_lp_mosek import solve_mosek
        #from solve_sparse_mosek import solve_mosek
        #from solve_pump_mosek import solve_mosek
        (x, fval, obj) = solve_mosek(in_file_name)
        x = np.array(x)
    elif solver == 'gurobi':
        from solve_lp_gurobi import solve_gurobi
        (x, fval) = solve_gurobi(in_file_name)
        x = np.array(x)
    else:
        print "solver %s is not implemented"%solver
    print "objective: %g"%fval
    print 'saving x.mat'
    sio.savemat('x.mat', {'x':x}) 
    as_mat = sio.loadmat(in_file_name)
    var_dict = pkl.loads(as_mat['var_dict_pkl'][0])
    dump_dict = {}
    for var_name, var in var_dict.iteritems():
        #if '<>' in var_name:
        components = select_from_solution(var, x)
        if np.min(np.array(components)) < -0.1:
            print var_name, components
            #pdb.set_trace()
        discrete = np.unravel_index(np.argmax(components), var['dims'])
        #print '%s = %s'%(var_name, discrete)
        #print components
        #print '*'*20
        dump_dict[var_name]={
            'components' : components,
            'discrete' : str(discrete)
        }
    if not out_file_name:
        out_file_name = '%s.json'%os.path.splitext(in_file_name)[0]
    if fval < -1e-3:    
        find_contributions_to_objective(obj, x, var_dict)
    export_json(out_file_name, dump_dict)
    pklfilename = '%s.pkl' % out_file_name
    with open(pklfilename, 'w') as f: 
        pkl.dump(dump_dict, f)
    
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        solve(sys.argv[1])
    elif len(sys.argv) > 2:
        solve(sys.argv[1], sys.argv[2])

    else:
        print ("please specify a file to solve")
        exit(1)