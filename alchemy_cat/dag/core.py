""" Main module containing Graph / Node classes """

from functools import reduce
import logging
import inspect
import copy
from multiprocessing.pool import Pool

from typing import Optional, Union, List, Callable, Dict, Tuple, Any

from alchemy_cat.dag.io import Input, Output
from alchemy_cat.dag.errors import PyungoError
from alchemy_cat.dag.utils import get_function_return_names, run_node
from alchemy_cat.dag.data import Data
from alchemy_cat.py_tools import Timer

logging.basicConfig()
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


def topological_sort(data):
    """ Topological sort algorithm

    Args:
        data (dict): dictionary representing dependencies
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
    extra_items_in_deps = reduce(set.union, data.values()) - set(data.keys())  # Add items without dependencies to data
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
        fct (Callable): The Python function attached to the node
        outputs (list): List of outputs (`Output` or `str`)
        args (list): Optional list of args
        kwargs (list): Optional list of kwargs

    Raises:
        PyungoError: In case inputs have the wrong type
    """

    def __init__(self, fct: Callable, outputs=None, args=None, kwargs=None, verbose=False, slim_names=None):
        self._fct = fct
        self._verbose = verbose
        self._slim_names = slim_names if slim_names else []

        self._inputs = []
        self._outputs = []
        self._fct_defaults = {}

        self._args = args if args else []
        self._kwargs = kwargs if kwargs else []

        # if inputs are None, we inspect the function signature
        if (not self._args) and (not self._kwargs):
            self._args = list(inspect.signature(self._fct).parameters.keys())
            if not self._args:
                raise PyungoError(f"Node's func {self._fct} must have input params")

        self._output_names = outputs if outputs else []

        # if inputs are None, we inspect the function code
        if not self._output_names:
            self._output_names = get_function_return_names(self._fct)

        self._process_inputs(self._args)
        self._process_inputs(self._kwargs, is_kwarg=True)

        self._process_fct_defaults()

        self._process_outputs(self._output_names)

        self._id = str(self)

    def __repr__(self):
        return 'Node(<{}>, {}, {})'.format(self.fct_name, self.input_names, self.output_names)

    def __call__(self, *args, **kwargs):
        """ run the function attached to the node, and store the result """
        if self._verbose:
            timer = Timer().start()
        res = self._fct(*args, **kwargs)
        if self._verbose:
            timer.close()
            LOGGER.info('Ran {} in {}'.format(self, timer))
        return res

    @property
    def id(self):
        """ return the unique id of the node """
        return self._id

    # @property
    # def inputs(self):
    #     " return inputs"
    #     return self._inputs

    @property
    def input_names(self):
        """ return a list of all input names """
        input_names = [i.name for i in self._inputs]
        return input_names

    # @property
    # def input_maps(self):
    #     """ return a list of all input maps """
    #     input_maps = [i.map for i in self._inputs]
    #     return input_maps

    @property
    def inputs_without_constants(self):
        """ return the list of inputs, when inputs are not constants """
        inputs = [i for i in self._inputs if not i.is_constant]
        return inputs

    # @property
    # def kwargs_name(self):
    #     """ return the list of kwargs """
    #     return [i.name for i in self._inputs if i.is_kwarg]

    # @property
    # def outputs(self):
    #     return self._outputs

    @property
    def output_names(self):
        """ return a list of output names """
        return [o.name for o in self._outputs]

    @property
    def nec_input_maps(self):
        """ return necessary input names """
        return [i.map for i in self._inputs if (not i.is_constant) and (i.name not in self._fct_defaults)]

    @property
    def opt_input_maps(self):
        """ return optional input names """
        return [i.map for i in self._inputs if (not i.is_constant) and (i.name in self._fct_defaults)]

    @property
    def fct_name(self):
        """ return the function name """
        if hasattr(self._fct, '__name__'):
            fct_name = self._fct.__name__
        else:
            fct_name = type(self._fct).__name__  # For fct is functor
        return fct_name

    def _process_inputs(self, inputs, is_kwarg=False):
        """ converter data passed to Input objects and store them """
        if isinstance(inputs, dict):
            inputs = [{k: v} for k, v in inputs.items()]

        for inp in inputs:
            if isinstance(inp, tuple):
                if len(inp) != 2:
                    raise PyungoError(f"Tuple input should like (name, map). However, get {inp}")
                new_input = Input(name=inp[0], map=inp[1], is_kwarg=is_kwarg)
            elif isinstance(inp, Input):
                new_input = inp
                new_input.is_kwarg = is_kwarg
            elif isinstance(inp, str):
                new_input = Input(name=inp, is_kwarg=is_kwarg)
            elif isinstance(inp, dict):
                if len(inp) != 1:
                    msg = ('dict inputs should have only one key '
                           'and cannot be empty')
                    raise PyungoError(msg)
                key = next(iter(inp))
                value = inp[key]
                new_input = Input(name=key, value=value, is_kwarg=is_kwarg)
            else:
                msg = 'inputs need to be of type tuple, Input, str or dict'
                raise PyungoError(msg)
            self._inputs.append(new_input)

    def _process_fct_defaults(self):
        """ read and store kwargs default values"""
        self._fct_defaults = {k: v.default for k, v in inspect.signature(self._fct).parameters.items()
                              if v.default is not inspect.Parameter.empty}

    def _process_outputs(self, outputs):
        """ converter data passed to Output objects and store them """
        for output in outputs:
            if isinstance(output, Output):
                new_output = output
            elif isinstance(output, str):
                new_output = Output(output)
            else:
                msg = 'outputs need to be of type Outputs or str'
                raise PyungoError(msg)
            self._outputs.append(new_output)

    def run_with_loaded_inputs(self):
        """ Run the node with the attached function and loaded input values """
        args = [i.value for i in self._inputs if not i.is_kwarg]
        kwargs = {i.name: i.value for i in self._inputs if i.is_kwarg}
        rslts = self(*args, **kwargs)
        # Save outputs
        if len(self._outputs) == 1:
            self._outputs[0].value = rslts
        else:
            for output, rslt in zip(self._outputs, rslts):
                output.value = rslt
        return rslts


class Graph:
    """ Graph object, collection of related nodes

    Args:
        pool_size (int): Size of the pool. 0 means don't use parallel.
        schema (dict): Optional JSON schema to validate inputs data
        verbosity (int): 0: No log output; 1: Only give graph level log; >1: Give graph and node level log.
        slim (bool): If True, use copy rather than deepcopy when setting value of Node's input.

    Raises:
        ImportError will raise in case parallelism is chosen and `multiprocess`
            not installed
    """

    def __init__(self, pool_size: int = 0, schema: dict = None, verbosity: int = 0, slim: bool = False):
        self._nodes = {}
        self._data = None
        self._schema = schema

        self.pool_size = pool_size
        self.verbosity = verbosity
        self.slim = slim

        self._dag = None

        self._dag_output_names = set()
        self._dag_nec_input_maps = set()
        self._dag_opt_input_maps = set()

    @property
    def data(self):
        """ return the data of the graph (inputs + outputs) """
        return self._data

    # @property
    # def sim_inputs(self):
    #     """ return input names (mapped) of every nodes need feeding from Data """
    #     inputs = []
    #     for node in self._nodes.values():
    #         inputs.extend([i.map for i in node.inputs_without_constants
    #                        if not i.is_kwarg])
    #     return inputs
    #
    # @property
    # def sim_kwargs(self):
    #     """ return kwarg names (mapped) of every nodes """
    #     kwargs = [k for node in self._nodes.values() for k in node.kwargs_name]
    #     return kwargs
    #
    # @property
    # def sim_outputs(self):
    #     """ return output names (mapped) of every nodes """
    #     outputs = []
    #     for node in self._nodes.values():
    #         outputs.extend([o.map for o in node.outputs])
    #     return outputs

    def _dependencies(self):
        """ return dependencies among the nodes """
        dep = {}
        for node in self._nodes.values():
            d = dep.setdefault(node.id, [])
            for inp in node.inputs_without_constants:
                for node2 in self._nodes.values():
                    if inp.map in node2.output_names:
                        d.append(node2.id)
        return dep

    @property
    def dependencies(self):
        """Return dependencies of each node"""
        return self._dependencies()

    @property
    def dag(self):
        """ If dag is not built, build dag. Return the ordered nodes graph """
        if self._dag is None:
            dag = []
            for node_ids in topological_sort(self._dependencies()):
                nodes = [self.get_node(node_id) for node_id in node_ids]
                dag.append(nodes)
            self._dag = dag
        return self._dag

    @property
    def ordered_nodes(self) -> List[Node]:
        """Same to the dag except returned nodes is 1-Dimension"""
        ordered_nodes = []
        for nodes in self.dag:
            ordered_nodes.extend(nodes)
        return ordered_nodes

    @property
    def deep_ordered_nodes(self):
        """Return all nodes in graph (include sub-graph's nodes) according to their running start order"""
        deep_ordered_nodes = []
        for node in self.ordered_nodes:
            deep_ordered_nodes.append(node)
            if isinstance(node._fct, Graph):
                deep_ordered_nodes.extend(node._fct.deep_ordered_nodes)
        return deep_ordered_nodes

    def deep_prefix_id_ordered_nodes(self, prefix: str= ''):
        """Return all (prefix_id, nodes) in graph (include sub-graph's nodes) according to their running start order"""
        ret = []
        for node in self.ordered_nodes:
            prefix_id = prefix + '.' + node.id if prefix else node.id
            ret.append((prefix_id, node))
            if isinstance(node._fct, Graph):
                ret.extend(node._fct.deep_prefix_id_ordered_nodes(prefix=prefix_id))
        return ret

    @staticmethod
    def run_node(node):
        """ run the node

        Args:
            node (Node): The node to run

        Returns:
            results (tuple): node id, node output values

        Raises:
            PyungoError
        """
        return node.id, node.run_with_loaded_inputs()

    def _register(self,
                  f: Union[Callable, type],
                  inputs: Optional[
                      Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                  outputs: Optional[List[Union[str, Output]]] = None,
                  args: Optional[
                      Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                  kwargs: Optional[
                      Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                  slim_names: Optional[List[str]] = None,
                  init: Optional[Dict[str, Any]] = None):
        """ get provided inputs, outputs, args, kwargs, slim_names, init if any and create a new node """
        if not callable(f):
            raise PyungoError(f"Registered function {f} should be callable")

        if (inputs is not None) and (args is not None):
            args = inputs + args
        elif inputs is not None:
            args = inputs

        # Instantiate f if f is functor
        if inspect.isclass(f):
            init = init if init is not None else {}
            f = f(**init)
        elif init is not None:
            raise PyungoError("Only functor can be initialized with 'init'")

        if isinstance(f, Graph):
            if args is not None:
                raise PyungoError(f"Node with Graph can only accept kwargs input. However, get args = {args}")

        # create and save the node to the graph
        node = Node(f, outputs, args, kwargs, True if self.verbosity > 1 else False, slim_names)
        # assume that we cannot have two nodes with the same output names
        diff = set(node.output_names) & self._dag_output_names
        if diff:
            raise PyungoError(f"Node {node} have repeated output names: {sorted(list(diff))}")

        # assume no self dependencies
        diff = set([inp.map for inp in node.inputs_without_constants]) & set(node.output_names)
        if diff:
            raise PyungoError(f"Node {node} have self dependence caused by the following inputs: {sorted(list(diff))}")

        self._dag_output_names |= set(node.output_names)
        self._dag_nec_input_maps |= set(node.nec_input_maps)
        self._dag_opt_input_maps |= set(node.opt_input_maps)
        self._dag_opt_input_maps -= self._dag_nec_input_maps

        self._nodes[node.id] = node
        self._dag = None

    def register(self,
                 inputs: Optional[
                     Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                 outputs: Optional[List[Union[str, Output]]] = None,
                 args: Optional[Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                 kwargs: Optional[
                     Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                 slim_names: Optional[List[str]] = None,
                 init: Optional[Dict[str, Any]] = None):
        """ register decorator

            Args:
                function : Python function attached to the node
                inputs (list): List of inputs (Input, str, or dict)
                outputs (list): List of outputs (Output or str)
                args (list): List of optional args
                kwargs (list): List of optional kwargs
                slim_names (list): List of args which should use copy rather than deepcopy
                init (dict): init dict for functor
        """

        def decorator(f):
            self._register(f, inputs, outputs, args, kwargs, slim_names, init)
            return f

        return decorator

    def add_node(self,
                 function: Union[Callable, type],
                 inputs: Optional[
                     Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                 outputs: Optional[List[Union[str, Output]]] = None,
                 args: Optional[Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                 kwargs: Optional[
                     Union[List[Union[str, Tuple[str, str], Input, Dict[str, Any]]], Dict[str, Any]]] = None,
                 slim_names: Optional[List[str]] = None,
                 init: Optional[Dict[str, Any]] = None):
        """ explicit method to add a node to the graph

        Args:
            function : Python function attached to the node
            inputs (list): List of inputs (Input, str, or dict)
            outputs (list): List of outputs (Output or str)
            args (list): List of optional args
            kwargs (list): List of optional kwargs
            slim_names (list): List of args which should use copy rather than deepcopy
            init (dict): init dict for functor
        """
        self._register(function, inputs, outputs, args, kwargs, slim_names, init)

    def get_node(self, id_: str) -> Node:
        """ get a node from its id """
        return self._nodes[id_]

    def deep_get_node(self, prefix_id: str):
        """ get a node by it's prefix_id"""
        graph = self
        ids = prefix_id.split('.')
        for id in ids[:-1]:
            graph = graph.get_node(id)._fct
            if not isinstance(graph, Graph):
                raise PyungoError(f"For node id = {id} in prefix_id = {prefix_id}, "
                                  f"it's fct has type {type(graph)} rather than Graph")
        return graph.get_node(ids[-1])

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

        if self.verbosity:
            timer = Timer().start()
            LOGGER.info('Starting calculation...')

        self._data = Data(data, self.slim)
        self._data.check_inputs(self._dag_nec_input_maps, self._dag_opt_input_maps, self._dag_output_names)

        def set_node_input_value(node, input, value):
            if (not self.slim) and (input.name not in node._slim_names):
                input.value = copy.deepcopy(value)
            else:
                input.value = value

        for nodes in self.dag:
            # loading node with inputs
            for node in nodes:
                for inp in node.inputs_without_constants:
                    if inp.map in self._data:
                        set_node_input_value(node, inp, self._data[inp.map])
                    else:
                        set_node_input_value(node, inp, node._fct_defaults[inp.name])

            # running nodes
            if self.pool_size:
                pool = Pool(self.pool_size)
                rslts = pool.map(run_node, nodes)  # ! The output of node's won't be copied back
                pool.close()
                pool.join()

                for node, rslt in zip(nodes, rslts):
                    if len(node._outputs) == 1:
                        node._outputs[0].value = rslt
                    else:
                        for output, r in zip(node._outputs, rslt):
                            output.value = r
            else:
                for node in nodes:
                    node.run_with_loaded_inputs()

            # save results
            for node in nodes:
                for output in node._outputs:
                    self._data[output.map] = output.value

        if self.verbosity:
            timer.close()
            LOGGER.info(f'Calculation finished in {timer}')

        last_node_outputs = self.dag[-1][-1]._outputs
        if len(last_node_outputs) == 1:
            return last_node_outputs[0].value
        else:
            return tuple(output.value for output in last_node_outputs)

    def __call__(self, *args, **kwargs):
        "Equal to graph.calculate(kwargs)"
        if args:
            raise PyungoError("Graph only receive keyword args which will be recognized as input name and value.")

        return self.calculate(kwargs)

    def __repr__(self):
        return f"Graph with {len(self._nodes)} nodes in " + \
               ("slim mode" if self.slim is True else "non-slim mode") + \
               '.'

    def __str__(self):
        ret = repr(self) + "\n"
        for prefix_id, node in self.deep_prefix_id_ordered_nodes():
            ret += f"{prefix_id}\n"
            if isinstance(node._fct, Graph):
                ret += "--> " + repr(node._fct) + "\n"
        return ret
