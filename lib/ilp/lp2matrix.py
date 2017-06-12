import numpy as np
import scipy.io as sio
import re
import pdb
import scipy.sparse as ss
# import LPRuntime as rt
import pickle as pkl


def vec_to_dense(dims, vals):
    sparse_mat = ss.coo_matrix((vals[:, 1], (vals[:, 0], np.zeros(vals.shape[0]))), shape=(dims[0], 1))
    mat = np.array(sparse_mat.todense())
    return mat[:, 0]


OBJECTIVE_AS_CONSTRAINT = True

class rt(object):
    pass


class LPCompiler(object):
    var_name_regex = re.compile('(?P<var>[^<>]*[^<>_])_?(?P<lbl><.*>)?')

    def __init__(self):
        self.total_var_length = 0
        self.const = {}
        self.var_dict = {}
        self.eq_count = 1 if OBJECTIVE_AS_CONSTRAINT else 0
        self.A = []
        self.b = []
        self.int_flag = []
        self.objective = []
        self.rt = rt()
        self.constr_labels = []

    def preallocateVarList(self, nVars):
        pass

    def addVariable(self, var_name, var_dims, kind='Var', **kwargs):
        if var_name in self.var_dict:
            print "error: %s is a duplicated variable"%var_name
        var_name_parts = LPCompiler.var_name_regex.match(var_name)
        new_var = {
            'start' : self.total_var_length,
            'dims'  : var_dims,
            'cumdims': np.cumprod([1]+var_dims),
            'length': np.prod(var_dims),
            'name'  : var_name,
            'idx'   : var_name_parts.group('var').split(','),
            'lbl'   : var_name_parts.group('lbl').split(','),
            'kind'  : kind,
            'min_value' : 0 if 'min_value' not in kwargs else kwargs['min_value']
        }
        self.var_dict[var_name] = new_var
        if kind == 'Param':
            for i in range(new_var['start'], new_var['start']+new_var['length']):
                self.int_flag.append([i, 1])
        self.total_var_length += new_var['length']

    def preallocateEqLists(self, nVars):
        pass

    def parse_sum_string(self, sum_string):
        sum_string_parts = sum_string.split(';')
        to_sum = {}
        for sum_part in sum_string_parts:
            if '=' not in sum_part:
                to_sum[sum_part.strip()]= ':' # i.e. sum over all elements
            else:
                var_val = sum_part.split('=')
                pre_process = re.sub(r'const_(\w+)', r"self.const['\1']", var_val[1])
                val = eval(pre_process)
                to_sum[var_val[0].strip()] = val
        return to_sum

    # LHS should be a list of 3-tuples
    def addEquality(self, LHS, RHS, label):
        if not isinstance(RHS, np.ndarray):
            RHS = np.array([RHS])
        n_eq = len(RHS)
        eq_offset = self.eq_count

        for eq, val in enumerate(RHS):
            if val != 0:
                self.b.append([eq+eq_offset, val])

        for term in LHS:
            coeff = term[0]
            var = self.var_dict[term[1]]
            to_sum = self.parse_sum_string(term[2])

            axis_ranges = [None]*len(var['dims'])
            no_sum_axis_indices = []
            sum_axis_indices = []
            for axis in range(len(var['dims'])):
                axis_name = var['idx'][axis]
                if axis_name in to_sum: # we are summing over this axis
                    sum_axis_indices.append(axis)
                    if to_sum[axis_name] is ':': # we are summing over all elements
                        axis_ranges[axis] = range(var['dims'][axis])
                    else: # we are summing specific elements
                        axis_ranges[axis] = to_sum[axis_name]
                else: # we are not summing over this axis
                    no_sum_axis_indices.append(axis)
                    axis_ranges[axis] = range(var['dims'][axis])

            # make the indices that will not be summed over
            no_sum_ranges = [axis_ranges[i] for i in no_sum_axis_indices]
            # magic trick:
            no_sum_indices = np.asarray(map(lambda x: x.flatten(), np.meshgrid(*no_sum_ranges)))
            assert(((n_eq == 1) or (no_sum_indices.size == n_eq))), "Dimension of un-summed indices does not match the expected number of equations"

            ranges_to_sum = [None]*len(var['dims'])
            for eq in range(n_eq):
                for i, axis in enumerate(no_sum_axis_indices):
                    ranges_to_sum[axis] = no_sum_indices[i,eq]
                for i, axis in enumerate(sum_axis_indices):
                    ranges_to_sum[axis] = axis_ranges[axis]
                # magic trick:
                sum_indices = np.asarray(map(lambda x: x.flatten(), np.meshgrid(*ranges_to_sum)))
                for s in range(sum_indices.shape[1]):
                    sp_i = eq + eq_offset
                    sp_j = var['start'] + np.sum(sum_indices[:,s]*var['cumdims'][0:-1])
                    sp_v = coeff
                    self.A.append([sp_i,sp_j,sp_v])
        self.constr_labels += [label]*n_eq          
        self.eq_count += n_eq

    def addObjective(self, var_string, indicator_string, local_vals=None):
        if local_vals is None:
            local_vals = locals()
        else:
            local_vals.update(locals())

        var = self.var_dict[var_string]
        n_axes = len(var['dims'])
        axis_ranges = [range(var['dims'][axis]) for axis in range(n_axes)]
        # magic trick:
        all_var_indices = np.asarray(map(lambda x: x.flatten(), np.meshgrid(*axis_ranges)))
        indices = []
        for i in range(all_var_indices.shape[1]):
            element_string = indicator_string
            # FIXME could we replace this for loop with a single regex?
            for v in range(n_axes):
                axis_var_value = all_var_indices[v,i] + var['min_value']
                element_string = re.sub('\[%s\]'%var['idx'][v], str(axis_var_value), element_string)

            # element = eval(element_string, globals(), local_vals)
            # if type(element) is bool:
            #     element = 0 if element else -1000
            exp_element = 1 if eval(element_string) else 0
            element = -1 if exp_element == 0 else np.log(exp_element)
            
            sp_i = var['start']+np.sum(all_var_indices[:,i]*var['cumdims'][0:-1])
            indices.append(sp_i)
            if element != 0:
                self.objective.append([sp_i,element])
        return indices
    
    def clearObjective(self):
        self.objective = []

    def make_final_A_b(self):
        A_vals = np.array(self.A)
        self.A_final = ss.csr_matrix((A_vals[:, 2], (A_vals[:, 0], A_vals[:, 1])),
                                      shape=(self.eq_count, self.total_var_length))
        self.b_final = vec_to_dense([self.eq_count], np.array(self.b))
        return self.A_final, self.b_final

    def make_final_obj(self):
        self.obj_final = vec_to_dense([self.total_var_length], np.array(self.objective))
        return self.obj_final

    def penalize_new_states(self):
        for mem in range(3):
            for head in range(2):
                for newHead in range(2):
                    v = self.var_dict['newState_%i_%i_<>' % (head,mem)]
                    self.objective.append((v['start']+newHead, -10 if newHead>head else 0))

    def link_to_next_block(self):
        for b in range(5):
            for otherBlock in range(5):
                v = self.var_dict['thenBlocks_%i_<>' % b]
                nextBlock = (b+1)%5
                self.objective.append((v['start']+otherBlock, 0 if otherBlock == nextBlock else 10))

    def convert_objective_to_constraint(self):
        for obj_el in self.objective:
            if np.abs(obj_el[1]) > 0.1:
                self.A.append([0, obj_el[0], 1])
        self.b.append([0, 0])
        #self.eq_count += 1
        self.objective = zip(*[range(self.total_var_length), 0*(np.random.rand(self.total_var_length) - 0.5)])


    def save_to_mat(self, args):
        if OBJECTIVE_AS_CONSTRAINT:
            self.convert_objective_to_constraint()
        #self.penalize_new_states()
        #self.link_to_next_block()
        
        if len(args) > 1:
            out_filename = args[1]
        else:
            out_filename = 'A_b_obj.mat'

        pickled_var_dict = pkl.dumps(self.var_dict)

        sio.savemat(out_filename, {
            'A' : {'values' : np.array(self.A), 'dims' : [self.eq_count, self.total_var_length] },
            'b' : {'values' : np.array(self.b), 'dims' : [self.eq_count] },
            'objective' : {'values' : np.array(self.objective), 'dims' : [self.total_var_length] },
            'lb' : {'values' : [], 'dims' : [self.total_var_length]},
            'ub_minus_1' : {'values' : [], 'dims' : [self.total_var_length]},
            'int_flag' : {'values' : np.array(self.int_flag), 'dims' : [self.total_var_length] },
            'constraintLabels' : self.constr_labels,
            'var_dict_pkl' : pickled_var_dict
        })
