import ast
import inspect

from alchemy_cat.dag.errors import PyungoError


def get_function_return_names(fct):
    """ Return variable name(s) or return statement of the given function """
    lines = inspect.getsourcelines(fct)
    outputs = None
    for line in lines[0][::-1]:
        stripped = line.strip()
        if 'def' in stripped:
            # NOTE: This work as def is a reserved keyword which will trigger
            # invalid syntax if misused
            msg = 'No return statement found in {}'
            raise PyungoError(msg.format(fct.__name__))
        ast_tree = ast.parse(stripped)
        for ast_node in ast.walk(ast_tree):
            if isinstance(ast_node, ast.Return):
                if isinstance(ast_node.value, ast.Name):
                    outputs = [ast_node.value.id]
                elif isinstance(ast_node.value, ast.Tuple):
                    outputs = [
                        elt.id for elt in ast_node.value.elts
                        if isinstance(elt, ast.Name)
                    ]
                else:
                    name = ast_node.value.__class__.__name__
                    msg = ('Variable name or Tuple of variable names are '
                           'expected, got {}'.format(name))
                    raise PyungoError(msg)
                break
        if outputs:
            break
    return outputs


def run_node(node):
    """Python multiprocessing works strangely in windows. The pool function needed to be
    defined globally

    Args:
        node (Node): Node to be called

    Returns:
        rslts: Node's call output
    """
    return node.run_with_loaded_inputs()
