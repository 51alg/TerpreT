#!/usr/bin/env python
'''
Usage:
    compile_ilp.py [options] MODEL HYPERS DATA [OUTDIR]

Options:
    -h --help                Show this screen.
    --train-batch NAME       Name of the data batch with training data.
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib", 'ilp'))

from docopt import docopt
import traceback
import pdb
import ast
import astunparse
import tptv1
import utils as u
import os
import unroller
import astunparse
import imp

from ilp_utils import LPWriter, CaseNode, SwitchGroup, Declaration, Constant
from lp2matrix import LPCompiler


lpw = LPWriter()


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def make_matlab_filename(filename):
    return filename.replace("_", "").replace(".py", ".m")

def parse_set_to(expr_node):
    assert isinstance(expr_node, ast.Expr), "set_to node should be Expr"
    call_node = u.cast(expr_node.value, ast.Call)
    attribute_node = u.cast(call_node.func, ast.Attribute)
    name_node = u.cast(attribute_node.value, ast.Name)
    lhs_var = name_node.id

    possible_attributes = ["set_to", "set_to_constant", "observe_value", "set_message"]
    assert attribute_node.attr in possible_attributes, "unexpected attribute " + ast.dump(attribute_node)

    op, args = u.parse_factor_expression(call_node.args[0])
    if attribute_node.attr == "set_to_constant":
        assert op is None, "set_to_constant isn't a constant"
        op = "SetToConstant"
    elif attribute_node.attr == "set_to":
        if op is None:
            op = "Copy"
    elif attribute_node.attr == "observe_value":
        assert op is None, "observe_value isn't a constant"
        op = "ObserveValue"

    return lhs_var, [op] + args 

class Factor(object):
    def __init__(self, expr_node, case_node):
        self.var_name, factor = parse_set_to(expr_node)
        self.f = factor[0]
        self.arg_names = factor[1:]
        self.context = case_node

    def resolve_variables(self):
        self.decl = self.context.resolve_variable(self.var_name)

        self.args = []
        for arg_name in self.arg_names:
            if isinstance(arg_name, int):
                self.args.append(arg_name)
            else:
                arg = self.context.resolve_variable(arg_name)
                self.args.append(arg)

    def is_observation(self):
        return self.f == "ObserveValue"

    def is_constant(self):
        return self.f == "SetToConstant"

    def __repr__(self):
        return "%s.set_to(%s(%s, %s))" % (self.var_name, self.f, self.arg_names,
                                          self.context.context_str())

    def scope(self):
        all_vars = [self.decl] + self.args
        return filter(lambda x: isinstance(x, Declaration), all_vars)

    def scope_names(self):
        return [decl.var for decl in self.scope()]
        # return [self.var_name] + self.arg_names

    def local_marginal(self):
        scope_str = ",".join([decl.name for decl in self.scope()])
        return "%s_<%s>" % (scope_str, self.context.context_str())


def make_switch_group(if_node, parent_case):
    cases_ast = u.if_and_or_else_blocks(if_node)

    switch_group = SwitchGroup(None, parent_case.num_switch_groups())
    switch_group.set_parent_case(parent_case)
    switch_group.var_name = None
    for if_node in cases_ast:
        compare_node = u.cast(if_node.test, ast.Compare)
        var_name, val = u.parse_compare(compare_node)

        if switch_group.var_name is None:
            switch_group.var_name = var_name
        else:
            assert var_name == switch_group.var_name, "if blocks must switch on same var"

        case_node = make_case_node(if_node.body, var_name, val)
        switch_group.add_case(val, case_node)

    return switch_group


def make_case_node(body, var_name, val):
    case_node = CaseNode(var_name, val)
    for ch in body:
        if isinstance(ch, ast.If):
            switch_group = make_switch_group(ch, case_node)
            case_node.add_switch_group(switch_group)

        elif isinstance(ch, ast.Assign):
            if u.is_constant_definition(ch):
                const = Constant(ch, case_node)
                case_node.add_constant(const)

            else:
                decl = Declaration(ch, case_node)
                case_node.add_declaration(decl)
                
        elif (isinstance(ch, ast.FunctionDef) and
              filter(lambda x: x.func.id=='Runtime',ch.decorator_list)):
              case_node.add_runtime_function(ch)
        
        elif isinstance(ch, ast.Expr):
            factor = Factor(ch, case_node)
            case_node.add_factor(factor)

        else:
            case_node.unhandled.append(ch)
            # print "unhandled", ast.dump(ch)

    return case_node


def print_switch_block_tree(body, level=0):
    if isinstance(body, list):
        for node in body:
            if isinstance(node, SwitchGroup):
                print level*4*" ", "// start group %s#%s" % (node.var_name, node.idx())
                print_switch_block_tree(node.cases.values(), level=level)
                print level*4*" ", "// end group %s#%s" % (node.var_name, node.idx())
            elif isinstance(node, CaseNode):
                print level*4*" ", "%s=%s:" % (node.var_name, node.val)
                print_switch_block_tree(node.declarations, level=level+1)
                print_switch_block_tree(node.factors, level=level+1)
                print_switch_block_tree(node.switch_groups, level=level+1)
            else:
                print level*4*" ", node
    else:
        assert False


def decls_in_case_node(case_node):
    results = case_node.declarations + []
    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            results.extend(decls_in_case_node(child_case_node))

    return results


def active_vars(case_node):
    factors = case_node.factors + []
    decls = set([])
    for factor in factors:
        for decl in factor.scope():
            decls.add(decl)
    decls |= set(case_node.declarations)

    for switch_group in case_node.switch_groups:
        decls.add(case_node.resolve_variable(switch_group.var_name))
        for child_case_node in switch_group.cases.values():
            decls |= active_vars(child_case_node)

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            decls -= set(child_case_node.declarations)

    return decls


def declare_local_marginals(case_node):
    vars = active_vars(case_node)
    #local_marginals = case_node.make_local_marginals(vars)
    case_node.make_local_marginals(vars)
    
    #for var_name in local_marginals:
    #    marg_name = local_marginals[var_name]
    #    lpw.add_variable(marg_name, case_node.resolve_variable(var_name).size)

    for var in vars:
        kind = var.kind if case_node.context_str() == '' else 'Var'
        lpw.add_variable(case_node.to_local_marginal(var), var.size, kind)

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            declare_local_marginals(child_case_node)


def factors_in_case_node(case_node):
    factors = case_node.factors + []
    for factor in factors:
        factor.resolve_variables()

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            factors.extend(factors_in_case_node(child_case_node))
    return factors


def used_vars_in_case_node(case_node):
    factors = case_node.factors + []
    vars = set([])
    for factor in factors:
        for var in factor.scope():
            vars.add(var)

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            vars |= used_vars_in_case_node(child_case_node)

    return vars


def make_cross_level_constraints(case_node):

    parent_mus = case_node.local_marginals
    for switch_group in case_node.switch_groups:
        for var in parent_mus:
            children_mus = []
            unused_vals = []
            for child_case_node in switch_group.cases.values():
                child_mus = child_case_node.local_marginals
                if var in child_mus:
                    children_mus.append(child_mus[var])
                else:
                    unused_vals.append(child_case_node.val)

            if len(children_mus) > 0:
                if len(unused_vals) > 0:
                    # Var is used in some children but not all. Need to handle unused cases.
                    # get gate marginal in parent context
                    switch_marg = case_node.get_local_marginal(switch_group.var_name)

                    # create ghost marginal
                    ghost_marginal = switch_group.ghost_local_marginal(var)
                    lpw.add_variable(ghost_marginal, case_node.resolve_variable(var).size)

                    children_mus.append(ghost_marginal)
                    entries = [
                        ("+1", ghost_marginal, var),
                        ("-1", switch_marg, "%s=[%s]" % (switch_group.var_name,
                                                         ",".join([str(s + lpw.zero_one_index_offset())
                                                                   for s in unused_vals])))
                    ]
                    lpw.add_equality(entries, "0", n_eq=1)

                # make cross-level consistency constraints
                entries = [
                    ("+1", parent_mus[var], "")
                    ]
                for child_mu in children_mus:
                    entries.append(("-1", child_mu, ""))
                sz = case_node.resolve_variable(var).size
                target = "np.zeros((%s))" % sz
                lpw.add_equality(entries, target, n_eq=case_node.eval_constant(sz))

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            make_cross_level_constraints(child_case_node)


def make_total_probability_constraints(case_node):
    if case_node.is_global_case():
        target_marginal = 1

        for decl in case_node.declarations:
            entries = [("+1", case_node.get_local_marginal(decl.name), decl.name)]
            target = "1"
            lpw.add_equality(entries, target, n_eq=1)

    else:
        target_marginal, target_value = case_node.gate_value_local_marginal_and_value()
        target_value = case_node.eval_constant(target_value)

        for var_name in case_node.local_marginals: #case_node.declarations:
            decl = case_node.resolve_variable(var_name)
            entries = [
                ("+1", case_node.local_marginals[decl.name], decl.name),
                ("-1", target_marginal, "%s=%s" % (case_node.var_name,
                                                   target_value + lpw.zero_one_index_offset()))
                ]
            target = "0"
            lpw.add_equality(entries, target, n_eq=1)

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            make_total_probability_constraints(child_case_node)


def declare_factor_local_marginals(case_node):
    for factor in case_node.factors:
        if factor.is_observation():  continue
        if factor.is_constant():  continue
        if len(factor.scope()) <= 1: continue

        lpw.add_variable(factor.local_marginal(),
                         ",".join([str(decl.size) for decl in factor.scope()]))

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            declare_factor_local_marginals(child_case_node)


def make_local_consistency_constraints(case_node):
    for factor in case_node.factors:
        if factor.is_observation():  continue
        if factor.is_constant():  continue
        if len(factor.scope()) <= 1: continue
        
        for decl in factor.scope():
            rest_of_scope_vars = [decl2.name for decl2 in factor.scope()]
            rest_of_scope_vars = filter(lambda x: x != decl.name, rest_of_scope_vars)
            entries = [("+1", factor.local_marginal(), ";".join(rest_of_scope_vars)),
                       ("-1", case_node.local_marginals[decl.name], "")]
            target = "np.zeros((%s))" % decl.size
            lpw.add_equality(entries, target, n_eq=factor.context.eval_constant(decl.size))

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            make_local_consistency_constraints(child_case_node)


def make_objective(case_node):
    for factor in case_node.factors:
        term1 = factor.local_marginal() #",".join(factor.scope_names())
        if factor.is_observation() or factor.is_constant():
            try:
                observed_value = int(factor.arg_names[0])
                #observed_value += lpw.zero_one_index_offset()
            except ValueError:
                assert False, "Can only observe integer values for now. Ask DT if you need this (it's easy-ish)."
            term2 = "[%s] == %s" % (factor.var_name, observed_value)
        else:
            term2 = "[%s] == self.rt.%s(%s)" % (factor.var_name, factor.f,
                                        ",".join(objective_arg_strings(factor)))#["[%s]" % arg_name for arg_name in factor.arg_names]))
        lpw.add_objective(term1, term2)

    for switch_group in case_node.switch_groups:
        for child_case_node in switch_group.cases.values():
            make_objective(child_case_node)

def objective_arg_strings(factor):
    obj_arg_strings = []
    for arg_name in factor.arg_names:
        # should allow named constants here too, ignore this for now
        if isinstance(arg_name, basestring):
            obj_arg_strings.append("[%s]" % arg_name)
        else:
            obj_arg_strings.append(str(arg_name))
    return obj_arg_strings


def validate_tree(case_node):
    for switch_group in case_node.switch_groups:
        switch_group.validate()
        for child_case_node in switch_group.cases.values():
            validate_tree(child_case_node)

def make_mat_file(model_filename):
    module_name = os.path.basename(model_filename).replace(".py", "")
    module = imp.load_source(module_name, model_filename)
    LP = module.makeLP()
    LP.save_to_mat([None,model_filename.replace(".py", ".mat")])


def compile_ilp(model_filename, hypers_filename, data_filename,
                train_batch, out_dir):
    (parsed_model, data, hypers, out_name) = u.read_inputs(model_filename,
                                                           hypers_filename,
                                                           data_filename,
                                                           train_batch)

    parsed_model = tptv1.translate_to_tptv1(parsed_model, data, hypers)

    #print astunparse.unparse(parsed_model)
    tree = unroller.unroll_and_flatten(parsed_model,
                                       do_checks=False,
                                       print_info=False)
    print astunparse.unparse(tree)
    global_case = make_case_node(tree.body, None, None)

    # print "Declarations"
    # global_decls = u.decls_in_case_node(global_case)
    # for decl in global_decls:
    #    print decl
    # print

    # print "Resolving variable uses..."
    factors = factors_in_case_node(global_case)
    for factor in factors:
        factor.resolve_variables()

    declare_local_marginals(global_case)
    declare_factor_local_marginals(global_case)
    make_cross_level_constraints(global_case)
    make_total_probability_constraints(global_case)
    make_local_consistency_constraints(global_case)
    make_objective(global_case)

    for const in global_case.constants:
        lpw.add_constant(const.name, const.value)

    for rtf in global_case.runtime_functions:
        lpw.add_runtime_function(rtf)

    result_filename = os.path.join(out_dir, '%s.py' % make_matlab_filename(out_name))
    with open(result_filename, 'w') as out_stream:
        lpw.dump(out_stream)
    make_mat_file(result_filename)
    
    return



if __name__ == "__main__":
    args = docopt(__doc__)

    source_filename = args['MODEL']
    hypers_filename = args['HYPERS']
    data_filename = args['DATA']
    out_dir = args.get('OUTDIR', None) or "compiled/ilp/"
    out_dir = os.path.join(out_dir, "")
    train_batch = args.get('--train-batch', 'train') or 'train'

    try:
        compile_ilp(source_filename, hypers_filename, data_filename,
                    train_batch, out_dir)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
