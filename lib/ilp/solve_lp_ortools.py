from ortools.linear_solver import pywraplp

import scipy.io as sio
import pdb
import scipy.sparse as ss
import numpy as np

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
    obj_dims, obj_vals = unpack(data["objective"])

    b = vec_to_dense(b_dims, b_vals)
    obj = vec_to_dense(obj_dims, obj_vals)

    A = ss.csr_matrix((A_vals[:, 2], (A_vals[:, 0], A_vals[:, 1])), shape=A_dims)

    return A_dims, A, b, obj


def solve_or(filename=None):
    A_dims, A, b, obj = load()
    num_constraints, num_vars = A_dims
    print "Loaded", A_dims

    solver = pywraplp.Solver('SolveLP', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    objective = solver.Objective()

    # Create variables and set objective
    vars = [[]] * num_vars
    for j in xrange(num_vars):
        vars[j] = solver.NumVar(0.0, solver.infinity(), "x_%s" % j)
        objective.SetCoefficient(vars[j], obj[j])
    objective.SetMaximization()

    # Create constraints
    constraints = []
    for i in xrange(num_constraints):
        le_constraint = solver.Constraint(-solver.infinity(), b[i])
        ge_constraint = solver.Constraint(-solver.infinity(), -b[i])

        js = np.nonzero(A[i, :])[1]
        for j in js:
            le_constraint.SetCoefficient(vars[j], A[i, j])
            ge_constraint.SetCoefficient(vars[j], -A[i, j])

        constraints.append(le_constraint)
        constraints.append(ge_constraint)

    # SOLVE!
    status = solver.Solve()

    solution_objective = 0
    for j in xrange(num_vars):
        solution_objective += vars[j].solution_value() * obj[j]

    if status == solver.OPTIMAL:
        print "OPTIMAL!"
    else:
        print "SUBOPTIMAL :("

    x_out = []
    for j in xrange(num_vars):
        xj = vars[j].solution_value()
        x_out[j] = xj
        if True or xj > 0:
            print "x_%s" % j, vars[j].solution_value()

    print "Objective =", solution_objective
    
    return solution_objective, xj

if __name__ == "__main__":
    import sys
    import pdb
    import traceback
    try:
        solve_or()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


