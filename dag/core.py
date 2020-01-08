""" Main module containing Graph / Node classes """

import uuid
import datetime as dt
from functools import reduce
import logging
import inspect

from pyungo.io import Input, Output, get_if_exists
from pyungo.errors import PyungoError
from pyungo.utils import get_function_return_names
from pyungo.data import Data


logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def topological_sort(data):
    """ Topological sort algorithm

    Args:
        data (dict): dictionnary representing dependencies
            Example: {'a': ['b', 'c']} node id 'a' depends on
            node id 'b' and 'c'

    Returns:
        ordered (list): list of list of node ids
            Example: [['a'], ['b', 'c'], ['d']]
            The sequence is representing the order to be run.
            The nested lists are node ids that can be run in parallel

    Raises:
        PyungoError: In case a cyclic dependency exists
    """
    for key in data:
        data[key] = set(data[key])
    for k, v in data.items():
        v.discard(k)  # ignore self dependencies
    extra_items_in_deps = reduce(set.union, data.values()) - set(data.keys())
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if not dep)
        if not ordered:
            break
        yield sorted(ordered)
        data = {item: (dep - ordered) for item, dep in data.items()
                if item not in ordered}
    if data:
        raise PyungoError('A cyclic dependency exists amongst {}'.format(data))


class Node:
    """ Node object (aka vertex in graph theory)

    Args:
        fct (function): The Python function attached to the node
        inputs (list): List of inputs (which can be `Input`, `str` or `dict`)
        outputs (list): List of outputs (`Output` or `str`)
        args (list): Optional list of args
        kwargs (list): Optional list of kwargs

    Raises:
        PyungoError: In case inputs have the wrong type
    """

    def __init__(self, fct, inputs, outputs, args=None, kwargs=None):
        self._id = str(uuid.uuid4())
        self._fct = fct
        self._inputs = []
        self._process_inputs(inputs)
        self._args = args if args else []
        self._process_inputs(self._args, is_arg=True)
        self._kwargs = kwargs if kwargs else []
        self._process_inputs(self._kwargs, is_kwarg=True)
        self._kwargs_default = {}
        self._process_kwargs(self._kwargs)
        self._outputs = []
        self._process_outputs(outputs)

    def __repr__(self):
        return 'Node({}, <{}>, {}, {})'.format(
            self._id, self._fct.__name__,
            self.input_names, self.output_names
        )

    def __call__(self, *args, **kwargs):
        """ run the function attached to the node, and store the result """
        t1 = dt.datetime.utcnow()
        res = self._fct(*args, **kwargs)
        t2 = dt.datetime.utcnow()
        LOGGER.info('Ran {} in {}'.format(self, t2-t1))
        # save results to outputs
        if len(self._outputs) == 1:
            self._outputs[0].value = res
        else:
            for i, out in enumerate(self._outputs):
                out.value = res[i]
        return res

    @property
    def id(self):
        """ return the unique id of the node """
        return self._id

    @property
    def input_names(self):
        """ return a list of all input names """
        input_names = [i.name for i in self._inputs]
        return input_names

    @property
    def inputs_without_constants(self):
        """ return the list of inputs, when inputs are not constants """
        inputs = [i for i in self._inputs if not i.is_constant]
        return inputs

    @property
    def kwargs(self):
        """ return the list of kwargs """
        return self._kwargs

    @property
    def outputs(self):
        return self._outputs

    @property
    def output_names(self):
        """ return a list of output names """
        return [o.name for o in self._outputs]

    @property
    def fct_name(self):
        """ return the function name """
        return self._fct.__name__

    def _process_inputs(self, inputs, is_arg=False, is_kwarg=False):
        """ converter data passed to Input objects and store them """
        # if inputs are None, we inspect the function signature
        if inputs is None:
            inputs = list(inspect.signature(self._fct).parameters.keys())
        for input_ in inputs:
            if isinstance(input_, Input):
                new_input = input_
            elif isinstance(input_, str):
                if is_arg:
                    new_input = Input.arg(input_)
                elif is_kwarg:
                    new_input = Input.kwarg(input_)
                else:
                    new_input = Input(input_)
            elif isinstance(input_, dict):
                if len(input_) != 1:
                    msg = ('dict inputs should have only one key '
                           'and cannot be empty')
                    raise PyungoError(msg)
                key = next(iter(input_))
                value = input_[key]
                new_input = Input.constant(key, value)
            else:
                msg = 'inputs need to be of type Input, str or dict'
                raise PyungoError(msg)
            self._inputs.append(new_input)

    def _process_kwargs(self, kwargs):
        """ read and store kwargs default values """
        kwarg_values = inspect.getargspec(self._fct).defaults
        if kwargs and kwarg_values:
            kwarg_names = (inspect.getargspec(self._fct)
                           .args[-len(kwarg_values):])
            self._kwargs_default = {k: v for k, v in
                                    zip(kwarg_names, kwarg_values)}

    def _process_outputs(self, outputs):
        """ converter data passed to Output objects and store them """
        if outputs is None:
            outputs = get_function_return_names(self._fct)
        for output in outputs:
            if isinstance(output, Output):
                new_output = output
            elif isinstance(output, str):
                new_output = Output(output)
            self._outputs.append(new_output)

    def set_value_to_input(self, input_name, value):
        """ set a value to the targeted input name

        Args:
            input_name (str): Name of the input
            value: value to be assigned to the input

        Raises:
            PyungoError: In case the input name is unknown
        """
        for input_ in self._inputs:
            if input_.name == input_name:
                input_.value = value
                return
        msg = 'input "{}" does not exist in this node'.format(input_name)
        raise PyungoError(msg)

    def run_with_loaded_inputs(self):
        """ Run the node with the attached function and loaded input values """
        args = [i.value for i in self._inputs
                if not i.is_arg and not i.is_kwarg]
        args.extend([i.value for i in self._inputs if i.is_arg])
        kwargs = {i.name: i.value for i in self._inputs if i.is_kwarg}
        return self(*args, **kwargs)


class Graph:
    """ Graph object, collection of related nodes

    Args:
        inputs (list): List of optional `Input` if defined separately
        outputs (list): List of optional `Output` if defined separately
        parallel (bool): Parallelism flag
        pool_size (int): Size of the pool in case parallelism is enabled
        schema (dict): Optional JSON schema to validate inputs data

    Raises:
        ImportError will raise in case parallelism is chosen and `multiprocess`
            not installed
    """
    def __init__(self, inputs=None, outputs=None, parallel=False, pool_size=2,
                 schema=None):
        self._nodes = {}
        self._data = None
        self._parallel = parallel
        self._pool_size = pool_size
        self._schema = schema
        self._sorted_dep = None
        self._inputs = {i.name: i for i in inputs} if inputs else None
        self._outputs = {o.name: o for o in outputs} if outputs else None

    @property
    def data(self):
        """ return the data of the graph (inputs + outputs) """
        return self._data

    @property
    def sim_inputs(self):
        """ return input names (mapped) of every nodes """
        inputs = []
        for node in self._nodes.values():
            inputs.extend([i.map for i in node.inputs_without_constants
                           if not i.is_kwarg])
        return inputs

    @property
    def sim_kwargs(self):
        """ return kwarg names (mapped) of every nodes """
        kwargs = [k for node in self._nodes.values() for k in node.kwargs]
        return kwargs

    @property
    def sim_outputs(self):
        """ return output names (mapped) of every nodes """
        outputs = []
        for node in self._nodes.values():
            outputs.extend([o.map for o in node.outputs])
        return outputs

    @property
    def dag(self):
        """ return the ordered nodes graph """
        ordered_nodes = []
        for node_ids in topological_sort(self._dependencies()):
            nodes = [self._get_node(node_id) for node_id in node_ids]
            ordered_nodes.append(nodes)
        return ordered_nodes

    @staticmethod
    def run_node(node):
        """ run the node

        Args:
            node (Node): The node to run

        Returns:
            results (tuple): node id, node output values
        """
        return (node.id, node.run_with_loaded_inputs())

    def _register(self, f, **kwargs):
        """ get provided inputs if anmy and create a new node """
        inputs = kwargs.get('inputs')
        outputs = kwargs.get('outputs')
        args_names = kwargs.get('args')
        kwargs_names = kwargs.get('kwargs')
        self._create_node(
            f, inputs, outputs, args_names, kwargs_names
        )

    def register(self, **kwargs):
        """ register decorator """
        def decorator(f):
            self._register(f, **kwargs)
            return f
        return decorator

    def add_node(self, function, **kwargs):
        """ explicit method to add a node to the graph

        Args:
            function (function): Python function attached to the node
            inputs (list): List of inputs (Input, str, or dict)
            outputs (list): List of outputs (Output or str)
            args (list): List of optional args
            kwargs (list): List of optional kwargs
        """
        self._register(function, **kwargs)

    def _create_node(self, fct, inputs, outputs, args_names, kwargs_names):
        """ create a save the node to the graph """
        inputs = get_if_exists(inputs, self._inputs)
        outputs = get_if_exists(outputs, self._outputs)
        node = Node(fct, inputs, outputs, args_names, kwargs_names)
        # assume that we cannot have two nodes with the same output names
        for n in self._nodes.values():
            for out_name in n.output_names:
                if out_name in node.output_names:
                    msg = '{} output already exist'.format(out_name)
                    raise PyungoError(msg)
        self._nodes[node.id] = node

    def _dependencies(self):
        """ return dependencies among the nodes """
        dep = {}
        for node in self._nodes.values():
            d = dep.setdefault(node.id, [])
            for inp in node.input_names:
                for node2 in self._nodes.values():
                    if inp in node2.output_names:
                        d.append(node2.id)
        return dep

    def _get_node(self, id_):
        """ get a node from its id """
        return self._nodes[id_]

    def _topological_sort(self):
        """ run topological sort algorithm """
        self._sorted_dep = list(topological_sort(self._dependencies()))

    def calculate(self, data):
        """ run graph calculations """
        # make sure data is valid when using schema
        if self._schema:
            try:
                import jsonschema
            except ImportError:
                msg = 'jsonschema package is needed for validating data'
                raise ImportError(msg)
            jsonschema.validate(instance=data, schema=self._schema)
        t1 = dt.datetime.utcnow()
        LOGGER.info('Starting calculation...')
        self._data = Data(data)
        self._data.check_inputs(self.sim_inputs, self.sim_outputs, self.sim_kwargs)
        if not self._sorted_dep:
            self._topological_sort()
        for items in self._sorted_dep:
            # loading node with inputs
            for item in items:
                node = self._get_node(item)
                inputs = [i for i in node.inputs_without_constants]
                for inp in inputs:
                    if (not inp.is_kwarg or
                            (inp.is_kwarg and inp.map in self._data._inputs)):
                        node.set_value_to_input(inp.name, self._data[inp.map])
                    else:
                        node.set_value_to_input(inp.name,
                                                node._kwargs_default[inp.name])

            # running nodes
            if self._parallel:
                try:
                    from multiprocess import Pool
                except ImportError:
                    msg = 'multiprocess package is needed for parralelism'
                    raise ImportError(msg)
                pool = Pool(self._pool_size)
                results = pool.map(
                    Graph.run_node,
                    [self._get_node(i) for i in items]
                )
                pool.close()
                pool.join()
                results = {k: v for k, v in results}
            else:
                results = {}
                for item in items:
                    node = self._get_node(item)
                    res = node.run_with_loaded_inputs()
                    results[node.id] = res
            # save results
            for item in items:
                node = self._get_node(item)
                res = results[node.id]
                if len(node.outputs) == 1:
                    self._data[node.outputs[0].map] = res
                else:
                    for i, out in enumerate(node.outputs):
                        self._data[out.map] = res[i]
        t2 = dt.datetime.utcnow()
        LOGGER.info('Calculation finished in {}'.format(t2-t1))
        return res
