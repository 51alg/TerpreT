import numpy as np
import re
import astunparse
import ast
import sys
import utils as u
import pdb

one_index_re = re.compile("[a-zA-Z0-9]+_(\d+)_<.+")
two_index_re = re.compile("[a-zA-Z0-9]+_(\d+)_(\d+)_<.+")
three_index_re = re.compile("[a-zA-Z0-9]+_(\d+)_(\d+)_(\d+)_<.+")

one_index_re2 = re.compile("[a-zA-Z0-9]+_(\d+)_?$")
two_index_re2 = re.compile("[a-zA-Z0-9]+_(\d+)_(\d+)_?$")
three_index_re2 = re.compile("[a-zA-Z0-9]+_(\d+)_(\d+)_(\d+)_?$")
shared_index_re2 = re.compile("\w+SH")


def index_dominates_old(index1, index2):
    if index1 is None: return False
    if index2 is None: return True

    if len(index1) != len(index2):
        return len(index1) < len(index2)

    for i1, i2 in zip(index1, index2):
        if i1 < i2: return True
        elif i1 > i2: return False

    return False


def index_dominates(index1, index2):
    if index1 is None: return False
    if index2 is None: return True

    if len(index1) == 1 and len(index2) > 1: return True
    if len(index2) == 1 and len(index1) > 1: return False

    if len(index1) != len(index2):
        return len(index1) < len(index2)

    for i1, i2 in zip(index1, index2):
        if i1 < i2: return True
        elif i1 > i2: return False

    return False


def index_dominates2(index1, index2):
    return not index_dominates(index1, index2)


def compare_indices(index1, index2):
    if index1 is None and index2 is None: return 0
    if index1 is None: return 1  # None is the worst
    if index2 is None: return -1  # None is the worst

    # length 1 comes first
    if len(index1) == 1 and len(index2) > 1: return -1
    if len(index2) == 1 and len(index1) > 1: return 1

    for j in xrange(np.minimum(len(index1), len(index2))):
        i1 = index1[j]
        i2 = index2[j]
        if i1 < i2: return -1
        elif i1 > i2: return 1
    if len(index1) < len(index2): return -1
    if len(index2) < len(index1): return 1
    return 0


class LPWriter(object):
    def __init__(self):
        self.equalities = []
        self.variables = []
        self.objectives = []
        self.messages = []
        self.constants = []
        self.global_constants = []
        self.runtime_function_declarations = []
        self.runtime_functions = []
        self.n_equations = 0
        self.include_waitbar = True
        self.message_sizes = {}

    def extract_indices(self, local_marginal):
        for pattern in [one_index_re, two_index_re, three_index_re]:
            m = pattern.match(local_marginal)
            if m is not None:
                return tuple([int(d) for d in m.groups()])
        return None

    def extract_indices2(self, init_local_marginal):
        local_marginal = init_local_marginal.split("<")[0]
        local_marginals = local_marginal.split(",")
        result = []
        for local_marginal in local_marginals:

            m = shared_index_re2.match(local_marginal)
            is_shared = (m is not None)
            for pattern in [one_index_re2, two_index_re2, three_index_re2]:
                m = pattern.match(local_marginal)
                
                if m is not None:
                    # print local_marginal, m.groups()
                    if not is_shared:
                        result.append(tuple([int(d) for d in m.groups()]))
                    else:
                        result.append(tuple([int(d)-1000 for d in m.groups()]))
            #if m is not None:
                #pdb.set_trace()
             #   result[-1] = tuple([-1000+x for x in result[-1]])
        #if len(local_marginals) > 1:
        #    print init_local_marginal, result
        #    pdb.set_trace()
        return result


    def dominant_index(self, indices):
        result = None
        for index in indices:
            if index_dominates(index, result):
                result = index

        return result

    def dominant_index2(self, indices):
        result = None
        for index in indices:
            if index_dominates(index, result):
                result = index
        return result

    def add_equality(self, entries, target, n_eq):
        entry_strs = []
        indices = []
        for entry in entries:
            coeff, local_marginal, sum_over = entry
            entry_strs.append("%s, '%s', '%s'" % (coeff, local_marginal, sum_over))
            # indices.append(self.extract_indices(local_marginal))
            indices.extend(self.extract_indices2(local_marginal))

        if False and len(entries) == 1:
            index = (-1, )  # np.inf, np.inf, np.inf)
        else:
            index = self.dominant_index(indices)

        arg1 = "[(%s)]" % ("),(".join(entry_strs))
        arg2 = target
        result = "LP.addEquality(%s, %s, '%s') # %s -> %s" % (arg1, arg2, str(indices), str(indices), str(index))
        result += "  # %s" % (str(index))
        self.equalities.append((index, n_eq, result))
        self.n_equations += n_eq
        return result

    def add_variable(self, local_marginal, var_sizes, var_kind='Var'):

        indices = self.extract_indices2(local_marginal)
        if indices is None:
            index = None
        else:
            index = self.dominant_index(indices)

        result = "LP.addVariable('%s', [%s], kind='%s', label='%s')  # %s" % (local_marginal, var_sizes, var_kind,str(indices),str(indices))
        self.variables.append((index, result))
        return result

    def add_objective(self, local_marginal, expression):
        result = "LP.addObjective('%s', '%s')" % (local_marginal, expression)
        self.objectives.append(result)
        return result

    def add_message(self, local_marginal, expression, message_name, decl_size, group):
        result = "message_indices['%s'] = LP.addObjective('%s', '%s', local_vals=local_vals)" % (message_name, local_marginal, expression)
        self.messages.append(result)
        self.message_sizes[message_name] = (group, decl_size)
        return result

    def add_constant(self, name, value):
        if name.startswith("const_"):
            name = name[len("const_"):]
        # Matlab needs constants defined in two places
        self.constants.append("LP.const['%s'] = %s;" % (name, value))
        self.global_constants.append("%s = %s;" % (name, value))

    def add_runtime_function(self, rtf):
        rtf.decorator_list = []
        self.runtime_function_declarations.append(astunparse.unparse(rtf))
        self.runtime_functions.append('LP.rt.%s = %s' % (rtf.name, rtf.name))

    def dump_boilerplate_top(self, indent=0):
        self.write("import numpy as np", indent=indent)
        self.write("from lp2matrix import LPCompiler", indent=indent)
        self.write("import sys", indent=indent)
        self.write("import time", indent=indent)

    def dump_message_sizes(self, indent=0):
        self.write("def messageSizes(local_vals=None):", indent=indent)
        indent += 1
        self.write("result = [{}, {}, {}]", indent=indent)
        for key, pair in self.message_sizes.iteritems():
            group, size = pair
            self.write("result[%s]['%s'] = %s" % (group, key, size), indent=indent)
        self.write("return result", indent=indent)
        self.write()

    def dump_make_LP_def(self, indent=0):
        self.write("def makeLP(local_vals=None):", indent=indent)
        self.write("LP = LPCompiler()", indent=indent + 1)
        self.write("def Copy(a): return a", indent = indent + 1)
        self.write("LP.rt.Copy = Copy", indent = indent + 1)

    def dump_boilerplate_bottom(self, indent=0):
        self.write("if __name__ == '__main__':", indent=indent)
        self.write('LP = makeLP()', indent=indent + 1)
        self.write('LP.save_to_mat(sys.argv)', indent=indent + 1)

    def zero_one_index_offset(self):
        """ This should return 1 if we expect 1-indexing, 0 if we expect 0-indexing"""
        return 0

    def dump_section(self, section_label, section, indent=0):
        self.write("# " + section_label, indent=indent)
        print_waitbar = self.include_waitbar and len(section) > 30
        if print_waitbar:
            self.write('t=time.time()', indent=indent)
        progress = 0
        progress_tick = int(len(section) / 30)
        for s in section:
            progress += 1
            if print_waitbar and progress % progress_tick == 0:
                self.write('print \'\\r  [{0:30s}] {1:.1f}%%\'.format(\'-\' * int(%g), %g / 30. * 100.),' % (progress/float(len(section)) * 30., progress/float(len(section)) * 30.), indent=indent) 

            self.write(s, indent=indent)
        self.write()
        if print_waitbar:
            self.write('print \'\\r  [{0:30s}] {1:.1f}%\'.format(\'-\' * int(30), 30 / 30. * 100.),', indent=indent)
            self.write('print " time for %s:", time.time()-t,"s"' % section_label, indent=indent)


    def sort_equalities(self):
        # this pre-sorting makes sure that the order of equalities with degenerate indices is always the same
        presorted = sorted(self.equalities)
        #presorted = self.equalities
        self.equalities = sorted(presorted, cmp=lambda x, y: compare_indices(x[0], y[0]))
        new_equalities = []
        num_equalities = 0
        for eq in self.equalities:
            new_equalities.append(eq[2])# + " n=%s" % num_equalities)
            num_equalities += eq[1]

        self.equalities = new_equalities

    def sort_variables(self):
        # this pre-sorting makes sure that the order of variables with degenerate indices is always the same
        presorted = sorted(self.variables)
        #presorted = self.variables
        self.variables = sorted(presorted, cmp=lambda x, y: compare_indices(x[0], y[0]))
        self.variables = [pair[1] for pair in self.variables]

    def dump(self, f=None, indent=0):
        if f is None:
            f = sys.stdout
        self.f = f

        self.dump_boilerplate_top(indent=indent)

        self.dump_section('Global Constants', self.global_constants, indent=indent)
        self.dump_section('Runtime Functions', self.runtime_function_declarations, indent=indent)

        #self.dump_message_sizes(indent=indent)
        self.dump_make_LP_def(indent=indent)

        self.write()

        self.sort_equalities()
        self.sort_variables()

        indent += 1

        self.dump_section('Constants', self.constants, indent=indent)
        self.dump_section('Runtime Functions', self.runtime_functions, indent=indent)
        self.dump_section('Declarations', self.variables, indent=indent)
        self.dump_section('Equalities', self.equalities, indent=indent)
        self.dump_section('Objectives', self.objectives, indent=indent)
        #self.write("message_indices = {}", indent=indent)
        #self.dump_section('Messages', self.messages, indent=indent)
        #self.write("return LP, message_indices", indent=indent)
        self.write("return LP", indent=indent)
        self.write()
        self.write()
        indent -= 1

        self.dump_boilerplate_bottom(indent=indent)

    def dump_objective(self, f=None, indent=0):
        if f is None:
            f = sys.stdout
        self.f = f
        self.dump_boilerplate_top(indent=indent)
        self.write("def makeObjective(LP, local_vals=None):", indent=indent)
        indent += 1
        self.write("LP.clearObjective()", indent=indent)
        self.dump_section('Objectives', self.objectives, indent=indent)
        self.write("message_indices = {}", indent=indent)
        self.dump_section('Messages', self.messages, indent=indent)
        self.write("return message_indices", indent=indent)

    def write(self, data="", indent=0):
        self.f.write(indent * "    " + data + "\n")


class CaseNode(object):
    def __init__(self, var_name, val):
        self.var_name = var_name
        self.val = val
        self.switch_groups = []
        self.parent_group = None
        self.factors = []
        self.declarations = []
        self.constants = []
        self.runtime_functions = []
        self.unhandled = []
        self.local_marginals = {}

    def children(self):
        return filter(lambda x: isinstance(x, CaseNode), self.body)

    def set_body(self, body):
        self.body = body

    def add_switch_group(self, group):
        self.switch_groups.append(group)

    def add_declaration(self, decl):
        self.declarations.append(decl)

    def add_constant(self, const):
        self.constants.append(const)
    
    def add_runtime_function(self, rtf):
        #rtf.decorations_list=[]
        self.runtime_functions.append(rtf)

    def is_observed(self, decl):
        for factor in self.factors:
            if factor.var_name == decl.name and factor.is_observation():
                return True, factor.arg_names[0]
        return False, None

    def is_constant(self, decl):
        for factor in self.factors:
            if factor.var_name == decl.name and factor.is_constant():
                return True, factor.arg_names[0]
        return False, None

    def add_factor(self, factor):
        self.factors.append(factor)

    def eval_constant(self, str_rep):
        literal_value = None
        try:
            literal_value = int(str_rep)
        except ValueError:
            for const in self.constants:
                if const.name == str_rep:
                    literal_value = const.value

        # Didn't find in local scope. Go up a level.
        if self.parent_group is not None and self.parent_group.parent_case is not None:
            literal_value = self.parent_group.parent_case.eval_constant(str_rep)

        assert literal_value is not None, "unable to eval constant " + str_rep

        return literal_value

    def resolve_variable(self, name):
        for decl in self.declarations:
            if decl.name == name:
                return decl

        # Didn't find in local scope. Go up a level.
        if self.parent_group is None or self.parent_group.parent_case is None:
            assert False, "Unable to resolve variable " + name

        return self.parent_group.parent_case.resolve_variable(name)

    def __repr__(self):
        return "CaseNode[%s]" % (self.context_str())

    def make_local_marginals(self, declarations):
        for decl in declarations:
            # name = "mu[%s]_%s" % (self.context_str(), decl.name)
            name = self.to_local_marginal(decl)
            self.local_marginals[decl.name] = name
        return self.local_marginals
        
    def to_local_marginal(self, declaration):
        return "%s_<%s>" % (declaration.name, self.context_str())

    def is_global_case(self):
        return self.parent_group is None

    def gate_value_local_marginal_and_value(self, val=None):
        if val is None:
            val = self.val
        return "%s_<%s>" % (self.var_name, self.ancestor_context_str()), val
        #return "mu[%s]_%s(%s)" % (self.ancestor_context_str(), self.var_name, self.val)

    def get_local_marginal(self, var_name):
        return self.local_marginals[var_name]

    def idx(self):
        if self.parent_group is None:
            return "X"
        else:
            return self.parent_group.idx()

    def ancestor_context_str(self):
        if self.parent_group is None:
            return ""
        elif self.parent_group.parent_case is None:
            return ""
        else:
            return self.parent_group.parent_case.context_str()

    def context_str(self, val=None, eval=False):
        if self.parent_group is None:
            return ""
        elif self.parent_group.parent_case is None:
            return ""
        else:
            ancestry_context_str = self.ancestor_context_str()
            if val is None:
                val = self.val
            cur_context_str = "%s=%s#%s" % (self.var_name, val, self.idx())
            if ancestry_context_str is "":
                return cur_context_str
            else:
                return "%s,%s" % (ancestry_context_str, cur_context_str)

    def num_switch_groups(self):
        return len(self.switch_groups)


class SwitchGroup(object):
    def __init__(self, var_name, idx_val):
        self.var_name = var_name
        self.cases = {}
        self.parent_case = None
        self.idx_val = idx_val

    def add_case(self, val, case):
        case.parent_group = self
        self.cases[val] = case

    def set_parent_case(self, case_node):
        self.parent_case = case_node

    def idx(self):
        if self.parent_case is None:
            assert False
        else:
            return self.parent_case.idx() + "," + str(self.idx_val)

    def ghost_local_marginal(self, var_name):
        context_str = self.parent_case.context_str() + "," + self.var_name
        return "%s_<%s=@#%s>" % (var_name, context_str, self.idx())

    def __repr__(self):
        return "SwitchGroup(%s#%s)" % (self.var_name, self.idx())

    def validate(self):
        switch_decl = self.parent_case.resolve_variable(self.var_name)
        num_cases = self.parent_case.eval_constant(switch_decl.size)
        cases_are_covered = np.zeros(num_cases, dtype=bool)
        for case in self.cases:
            case = self.parent_case.eval_constant(case)
            assert case < num_cases, "switch case out of bounds %s: %s (size=%s)" % (self, case, num_cases)
            cases_are_covered[case] = True

        assert np.all(cases_are_covered), "not all cases covered for %s: %s" % (self, np.nonzero(1 - cases_are_covered))


class Declaration(object):
    def __init__(self, assign_node, case_node):
        self.parse(assign_node)
        self.context = case_node
        #self.is_observed, self.observed_value = case_node.is_observed(self)
        #self.is_constant, self.constant_value = case_node.is_constant(self)

    def __repr__(self):
        return "Declaration(%s, %s, %s, %s)" % (self.name, self.kind, self.size,
                                                self.context.context_str())

    def parse(self, assign_node):
        if len(assign_node.targets) > 1:  return False

        if u.is_constant_definition(assign_node):
            return None

        self.name = assign_node.targets[0].id

        rhs = assign_node.value
        if isinstance(rhs, ast.Call):
            call_node = u.cast(rhs, ast.Call)
            self.parse_call(call_node)
            self.array_size = None

        elif isinstance(rhs, ast.Subscript):
            subscript_node = u.cast(rhs, ast.Subscript)
            call_node = u.cast(subscript_node.value, ast.Call)
            self.parse_call(call_node)
            self.array_size = u.get_index(subscript_node)

    def parse_call(self, call_node):
        self.kind = call_node.func.id
        self.size = u.name_or_number(call_node.args[0])
        assert len(call_node.args) <= 1, "shouldn't have more than 1 arg to decl (how to define consts changed)"


class Constant(object):
    def __init__(self, assign_node, case_node):
        self.parse(assign_node)
        self.context = case_node

    def __repr__(self):
        return "Constant(%s, %s, %s)" % (self.name, self.value,
                                         self.context.context_str())

    def parse(self, assign_node):
        assert len(assign_node.targets) == 1

        self.name = assign_node.targets[0].id

        self.value = assign_node.value.n
