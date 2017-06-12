import numpy as np
import scipy.io as sio
import scipy.sparse as ss
import pdb
import sys,time

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

    A = ss.csc_matrix((A_vals[:, 2], (A_vals[:, 0], A_vals[:, 1])), shape=A_dims)
    
    # mask = np.array((A[0,:]==0).todense())[0]
    # A = A[1:,mask]
    # b = b[1:]
    # newintflag = np.zeros(len(obj))
    # newintflag[np.array(list(set(intflag)))]=1
    # newintflag = newintflag[mask]
    # newintflag = np.nonzero(newintflag)[0]
    # intflag = newintflag.tolist()
    # obj = obj[mask]
    # A_dims = np.array(np.shape(A))

    return A_dims, A, b, obj, set(intflag)

def print_column_entries(row_letter, col_idx, column_entries, outfile):
    n_double_rows = len(column_entries) / 4 # deliberate integer division
    for i in range(n_double_rows):
        print >> outfile, '    {0:1s}{1:7s}  {2:8s}  {3:12s}   {4:8s}  {5:12s}'.format(row_letter,str(col_idx), 
        column_entries[4*i], column_entries[4*i+1], column_entries[4*i+2], column_entries[4*i+3])
    if len(column_entries) % 4 is not 0:
        print >> outfile, '    {0:1s}{1:7s}  {2:8s}  {3:12s}'.format(row_letter,str(col_idx),
        column_entries[-2], column_entries[-1])
    
def to_mps(filename, outfile):
    print >> outfile, "NAME", "myLP.mps"
    
    print >> outfile, "OBJSENSE"
    print >> outfile, " MAX" 
    
    A_dims, A, b, obj,intflag = load(filename)

    # ROWS
    print >> outfile, "ROWS"
    # objective
    print >> outfile, " N OBJ"
    # constraints
    constraint_start_time = time.time()
    for i in range(A_dims[0]):
        if i % 1000 == 0:
            print >> sys.stderr, '\r  [{0:30s}] {1:.1f}%'.format('-' * int(i/float(A_dims[0]) * 30), i/float(A_dims[0]) * 100.),
        print >> outfile, " E  C%i" % i
    print >> sys.stderr, '\r  [{0:30s}] {1:.1f}%'.format('-' * 30, 100.),
    print >> sys.stderr, ' time for rows: %g s' % (time.time()-constraint_start_time)
                
    # COLUMNS
    print >> outfile, "COLUMNS"
    M_count = 0
    for j in range(A_dims[1]):
        if j % 1000 == 0:
            print >> sys.stderr, '\r  [{0:30s}] {1:.1f}%'.format('-' * int(j/float(A_dims[1]) * 30), j/float(A_dims[1]) * 100.),
        (row,_,val) = ss.find(A[:,j])
        row_names = map(lambda x : 'C%i' % x, row)
        column_entries = [str(v) for pair in zip(row_names,val) for v in pair]
        if np.abs(obj[j]) > 1e-5:
            column_entries.append('OBJ')
            column_entries.append(str(obj[j]))
            
        # is_int = (j in intflag)
        # if is_int:
        #     print "    M{0:7s}  'MARKER'                 'INTORG'".format(str(M_count))
        #     M_count += 1
        print_column_entries('V', j, column_entries, outfile)
        # if is_int:
        #     print "    M{0:7s}  'MARKER'                 'INTEND'".format(str(M_count))
        #     M_count += 1
    print >> sys.stderr, '\r  [{0:30s}] {1:.1f}%'.format('-' * 30, 100.),        
    print >> sys.stderr, ' time for columns: %g s' % (time.time()-constraint_start_time)
    
    # RHS
    print >> outfile, "RHS"
    (_,row,val) = ss.find(b)
    row_names = map(lambda x : 'C%i' % x, row)
    column_entries = [str(v) for pair in zip(row_names,val) for v in pair]
    print_column_entries('R', 1, column_entries, outfile)
    
    # BOUNDS
    print >> outfile, "BOUNDS"
    #for j in range(A_dims[1]):
    for j in intflag:
    #    if j in intflag:
        print >> outfile, ' BV B{0:7s}  V{1:7s}  0'.format('1', str(j))
    #    else:
    #        print >> outfile, ' LO B{0:7s}  V{1:7s}  -1000'.format('1', str(j))
           
        
        
    print >> outfile, "ENDATA"

if __name__ == '__main__':     
    to_mps(sys.argv[1], sys.stdout)         