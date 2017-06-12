import ast
import sys

def error(msg, node):
    '''
    Print an error that includes line and column information.
    '''
    msg = '\n'.join(['    ' + line for line in msg.split('\n')])
    lineno = node.lineno if hasattr(node, 'lineno') else 0
    col_offset = node.col_offset if hasattr(node, 'col_offset') else 0
    print >> sys.stderr, "Error (line {}, col {}):\n{}\n".format(lineno, col_offset, msg)

def fatal_error(msg, node):
    '''
    Print an error message and then exit with a negative status.
    '''
    error(msg, node)
    raise RuntimeError('Fatal error during compilation.')

def check_type(node, type_):
    if not isinstance(node, type_):
        msg = "Expected a {} but got a {}.\n{}".format(type_.__name__, type(node).__name__, ast.dump(node))
        fatal_error(msg, node)
