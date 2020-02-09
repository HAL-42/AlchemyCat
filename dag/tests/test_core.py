import pytest

from alchemy_cat.dag.core import Graph, PyungoError
from alchemy_cat.dag.io import Input, Output


def test_simple():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['d', 'a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(inputs=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 10.

    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5

    # make sure it is indepodent
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5


def test_constant_inputs():
    graph = Graph()

    @graph.register(inputs=[{'a': 2}, {'b': 3}], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['d', 'a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(inputs=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 10.

    res = graph.calculate(data={'a' :2})
    assert res == -1.5
    assert graph.data['e'] == -1.5

    # make sure it is indepodent
    res = graph.calculate(data={'a': 2})
    assert res == -1.5
    assert graph.data['e'] == -1.5


def test_slim_graph():
    import numpy as np
    graph = Graph(slim=True)

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['d', 'a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(inputs=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 10.

    res = graph.calculate(data={'a': np.ones((2,2)) * 2., 'b': np.ones((2,2)) * 3.})
    assert (res == np.ones((2,2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2,2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) == id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) == id(graph.ordered_nodes[2]._inputs[0].value)

    res = graph.calculate(data={'a': np.ones((2,2)) * 2., 'b': np.ones((2,2)) * 3.})
    assert (res == np.ones((2,2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2,2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) == id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) == id(graph.ordered_nodes[2]._inputs[0].value)


def test_not_slim_graph():
    import numpy as np
    graph = Graph(slim=False)

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['d', 'a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(inputs=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 10.

    res = graph.calculate(data={'a': np.ones((2,2)) * 2., 'b': np.ones((2,2)) * 3.})
    assert (res == np.ones((2,2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2,2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) != id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) != id(graph.ordered_nodes[2]._inputs[0].value)

    res = graph.calculate(data={'a': np.ones((2,2)) * 2., 'b': np.ones((2,2)) * 3.})
    assert (res == np.ones((2,2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2,2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) != id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) != id(graph.ordered_nodes[2]._inputs[0].value)


def test_node_slim_graph():
    import numpy as np
    graph = Graph(slim=False)

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['d', 'a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(inputs=['c'], outputs=['d'], slim_names=['c'])
    def f_my_function2(c):
        return c / 10.

    res = graph.calculate(data={'a': np.ones((2,2)) * 2., 'b': np.ones((2,2)) * 3.})
    assert (res == np.ones((2,2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2,2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) == id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) != id(graph.ordered_nodes[2]._inputs[0].value)

    res = graph.calculate(data={'a': np.ones((2,2)) * 2., 'b': np.ones((2,2)) * 3.})
    assert (res == np.ones((2,2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2,2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) == id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) != id(graph.ordered_nodes[2]._inputs[0].value)


def test_functor():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    class f_my_function(object):
        factor = 1

        def __init__(self):
            self.dummy = 100

        def __call__(self, a1, a2):
            return a1 + a2 + self.factor

    @graph.register(inputs=['d', 'a'], outputs=['e'], init={'add_or_sub': 'add', 'factor': 2})
    class f_my_function2(object):
        def __init__(self, add_or_sub = 'add', factor=1):
            if add_or_sub == 'add':
                self.func = lambda x, y: x + y
            else:
                self.func = lambda x, y: x - y

            self.factor = factor

        def generate_constant(self):
            return 1

        def __call__(self, a1, a2):
            return (self.func(a1, a2) + self.generate_constant()) * self.factor


    @graph.register(inputs=['c', 'b'], outputs=['d'], init={'add_or_sub': 'sub'})
    class f_my_function1(object):
        def __init__(self, add_or_sub = 'add', factor=1):
            if add_or_sub == 'add':
                self.func = lambda x, y: x + y
            else:
                self.func = lambda x, y: x - y

            self.factor = factor

        def generate_constant(self):
            return 1

        def __call__(self, a1, a2):
            return (self.func(a1, a2) + self.generate_constant()) * self.factor

    res = graph.calculate(data={'a': 2, 'b': 3})
    assert graph.data['c'] == 6
    assert graph.data['d'] == 4
    assert res == 14
    assert graph.data['e'] == 14

    # make sure it is independent
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert graph.data['c'] == 6
    assert graph.data['d'] == 4
    assert res == 14
    assert graph.data['e'] == 14


def test_simple_without_decorator():
    graph = Graph()

    def f_my_function(a, b):
        return a + b

    def f_my_function3(d, a):
        return d - a

    def f_my_function2(c):
        return c / 10.

    graph.add_node(f_my_function, inputs=['a', 'b'], outputs=['c'])
    graph.add_node(f_my_function3, inputs=['d', 'a'], outputs=['e'])
    graph.add_node(f_my_function2, inputs=['c'], outputs=['d'])

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert res == -1.5
    assert graph.data['e'] == -1.5


def par_f_my_function(a, b):
        return a + b

def par_f_my_function3(d, a):
    return d - a

def par_f_my_function2(c):
    return c / 10.

def test_simple_parralel():
    """ TODO: We could mock and make sure things are called correctly """

    graph = Graph(pool_size=2)

    graph.add_node(par_f_my_function, inputs=['a', 'b'], outputs=['c'])
    graph.add_node(par_f_my_function3, inputs=['d', 'a'], outputs=['e'])
    graph.add_node(par_f_my_function2, inputs=['c'], outputs=['d'])
    graph.add_node(par_f_my_function2, inputs=['c'], outputs=['f'])
    graph.add_node(par_f_my_function2, inputs=['c'], outputs=['g'])

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert res == -1.5


def test_multiple_outputs():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c', 'd'])
    def f_my_function(a, b):
        return a + b, 2 * b

    @graph.register(inputs=['c', 'd'], outputs=['e'])
    def f_my_function2(c, d):
        return c + d

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert res == 11
    assert graph.data['e'] == 11


def test_same_output_names():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=['c'], outputs=['c'])
        def f_my_function2(c):
            return c / 10
    
    assert 'c output already exist' in str(err.value)


def test_missing_input():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6})
    
    assert "The following inputs are needed: ['b']" in str(err.value)


def test_inputs_not_used():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6, 'b': 4, 'e': 7})
    
    assert "The following inputs are not used by the model: ['e']" in str(err.value)


def test_inputs_collision():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6, 'b': 4, 'c': 7})
    
    assert "The following inputs are already used in the model: ['c']" in str(err.value)


def test_circular_dependency():
    graph = Graph()

    @graph.register(inputs=['a', 'b', 'd'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 2.

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6, 'b': 4})

    assert "A cyclic dependency exists amongst" in str(err.value)


def test_iterable_on_single_output():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return list(range(a)) + [b]

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert res == [0, 1, 3]
    assert graph.data['c'] == [0, 1, 3]


def test_multiple_outputs_with_iterable():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c', 'd'])
    def f_my_function(a, b):
        return list(range(a)) + [b], b * 10

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert isinstance(res, tuple) is True
    assert graph.data['c'] == [0, 1, 3]
    assert graph.data['d'] == 30
    assert res[0] == [0, 1, 3]
    assert res[1] == 30


def test_args_kwargs():
    graph = Graph()

    @graph.register(
        inputs=['a', 'b'],
        args=['c'],
        kwargs=['d'],
        outputs=['e']
    )
    def f_my_function(a, b, *args, **kwargs):
        return a + b + args[0] + kwargs['d']

    res = graph.calculate(data={'a': 2, 'b': 3, 'c': 4, 'd': 5})

    assert res == 14
    assert graph.data['e'] == 14


def test_constant_args_kwargs():
    graph = Graph()

    @graph.register(
        inputs=['a', 'b'],
        args=['c', {'cc': 6}],
        kwargs=['d', {'dc': 7}],
        outputs=['e']
    )
    def f_my_function(a, b, *args, **kwargs):
        return a + b + args[0] + args[1] + kwargs['d'] + kwargs['dc']

    res = graph.calculate(data={'a': 2, 'b': 3, 'c': 4, 'd': 5})

    assert res == 27
    assert graph.data['e'] == 27


def test_constant_dict_kwargs():
    graph = Graph()

    @graph.register(
        inputs=['a', 'b'],
        args=['c'],
        kwargs={'d': 5, 'dc': 6},
        outputs=['e']
    )
    def f_my_function(a, b, *args, **kwargs):
        return a + b + args[0] + kwargs['d'] + kwargs['dc']

    res = graph.calculate(data={'a': 2, 'b': 3, 'c': 4})

    assert res == 20
    assert graph.data['e'] == 20



def test_diff_input_function_arg_name():
    graph = Graph()

    @graph.register(
        inputs=['a_diff', 'b_diff'],
        args=['c_diff'],
        kwargs=['d'],
        outputs=['e_diff']
    )
    def f_my_function(a, b, *args, **kwargs):
        return a + b + args[0] + kwargs['d']

    res = graph.calculate(data={'a_diff': 2, 'b_diff': 3, 'c_diff': 4, 'd': 5})

    assert res == 14
    assert graph.data['e_diff'] == 14

def test_dag_pretty_print():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['d', 'a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(inputs=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 10.

    expected = ['f_my_function', 'f_my_function2', 'f_my_function3']
    dag = graph.dag
    for i, fct_name in enumerate(expected):
        assert dag[i][0].fct_name == fct_name


def test_passing_data_to_node_definition():

    graph = Graph()

    @graph.register(inputs=['a', {'b': 2}], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    res = graph.calculate(data={'a': 5})
    assert res == 7


def test_wrong_input_type():

    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=['a', {'b'}], outputs=['c'])
        def f_my_function(a, b):
            return a + b

    assert "inputs need to be of type Input, str or dict" in str(err.value)


def test_empty_input_dict():

    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=['a', {}], outputs=['c'])
        def f_my_function(a, b):
            return a + b

    assert "dict inputs should have only one key and cannot be empty" in str(err.value)


def test_multiple_keys_input_dict():

    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=['a', {'b': 1, 'c': 2}], outputs=['c'])
        def f_my_function(a, b):
            return a + b

    assert "dict inputs should have only one key and cannot be empty" in str(err.value)


def test_Input_type_input():
    graph = Graph()

    @graph.register(
        inputs=[Input(name='a'), 'b'],
        outputs=['c']
    )
    def f_my_function(a, b):
        return a + b

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert res == 5


@pytest.mark.skip("Don't Support Contract Now")
def test_contract_inputs():
    from contracts import ContractNotRespected
    graph = Graph()
    @graph.register(
        inputs=[Input(name='a', contract='int,>0'), 'b'],
        outputs=['c']
    )
    def f_my_function(a, b):
        return a + b
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == 5
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == 5
    with pytest.raises(ContractNotRespected) as err:
        res = graph.calculate(data={'a': -2, 'b': 3})
    assert "Condition -2 > 0 not respected" in str(err.value)


@pytest.mark.skip("Don't Support Contract Now")
def test_contract_outputs():
    from contracts import ContractNotRespected
    graph = Graph()
    @graph.register(
        inputs=['a', 'b'],
        outputs=[Output('c', contract='int,>0')]
    )
    def f_my_function(a, b):
        return a + b
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == 5
    with pytest.raises(ContractNotRespected) as err:
        res = graph.calculate(data={'a': -4, 'b': 3})
    assert "Condition -1 > 0 not respected" in str(err.value)


def test_provide_inputs_outputs():

    inputs = [Input('a'), Input('b')]
    outputs = [Output('c')]

    graph = Graph(inputs=inputs, outputs=outputs)

    @graph.register(
        inputs=['a', 'b'],
        outputs=['c']
    )
    def f_my_function(a, b):
        return a + b

    @graph.register(
        inputs=['a', 'c'],
        outputs=['d']
    )
    def f_my_function1(a, c):
        return a - c

    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -3
    assert inputs[0].value == 2
    assert inputs[1].value == 3
    assert outputs[0].value == 5


def test_provide_args_kwargs():

    inputs = [Input('b'), Input('c')]
    outputs = [Output('c')]

    graph = Graph(inputs=inputs, outputs=outputs)

    @graph.register(
        inputs=['a'],
        args=['b'],
        outputs=['c']
    )
    def f_my_function(a, *b):
        return a + b[0]

    @graph.register(
        inputs=['a'],
        kwargs=['c'],
        outputs=['d']
    )
    def f_my_function1(a, c=100):
        return a - c

    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -3
    assert inputs[0].value == 3
    assert inputs[1].value == 5
    assert outputs[0].value == 5


def test_provide_inputs_outputs_already_defined():

    inputs = [Input('a'), Input('b')]
    outputs = [Output('c')]

    graph = Graph(inputs=inputs, outputs=outputs)

    with pytest.raises(TypeError) as err:
        @graph.register(
            inputs=['a', 'b'],
            outputs=[Output('c')]
        )
        def f_my_function(a, b):
            return a + b

    msg = "You cannot use Input / Output in a Node if already defined"
    assert msg in str(err.value)


def test_map():

    graph = Graph()

    @graph.register(
        inputs=[Input('a', map='q'), Input('b', map='w')],
        outputs=[Output('c', map='e')]
    )
    def f_my_function(a, b):
        return a + b

    res = graph.calculate(data={'q': 2, 'w': 3})
    assert res == 5
    assert graph.data['e'] == 5


def test_schema():

    from jsonschema import ValidationError

    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    }

    graph = Graph(schema=schema)

    @graph.register(
        inputs=['a', 'b'],
        outputs=['c']
    )
    def f_my_function(a, b):
        return a + b

    with pytest.raises(ValidationError) as err:
        graph.calculate(data={'a': 1, 'b': '2'})

    msg = "'2' is not of type 'number'"
    assert msg in str(err.value)

    res = graph.calculate(data={'a': 1, 'b': 2})
    assert res == 3


def test_optional_kwargs_without_feed():
    graph = Graph()

    @graph.register(inputs=['a'], kwargs=['b'], outputs=['c'])
    def f(a, b=2):
        return a + b

    res = graph.calculate(data={'a': 1})

    assert res == 3
    assert graph.data['c'] == 3


def test_optional_kwargs_feed_by_input():
    graph = Graph()

    @graph.register(inputs=['a'], kwargs=['b'], outputs=['c'])
    def f(a, b=2):
        return a + b

    res = graph.calculate(data={'a': 1, 'b': 3})

    assert res == 4
    assert graph.data['c'] == 4


def test_optional_kwargs_feed_by_output():
    graph = Graph()

    @graph.register(inputs=['a'], kwargs=['b'], outputs=['c'])
    def f(a, b):
        return a + b

    @graph.register(inputs=['c'], kwargs=['c'], outputs=['d'])
    def f1(a, c=5):
        return a + c

    res = graph.calculate(data={'a': 1, 'b': 3})

    assert res == 8
    assert graph.data['c'] == 4


def test_no_explicit_inputs_outputs_simple():
    graph = Graph()

    @graph.register()
    def f(a, b):
        c = a + b
        return c

    res = graph.calculate(data={'a': 1, 'b': 2})

    assert res == 3
    assert graph.data['c'] == 3


def test_no_explicit_inputs_outputs_tuple():
    graph = Graph()

    @graph.register()
    def f(a, b, c, d):
        e = a + b
        f = c - d
        return e, f

    res = graph.calculate(data={'a': 1, 'b': 2, 'c': 3, 'd': 4})

    assert res == (3, -1)
    assert graph.data['e'] == 3
    assert graph.data['f'] == -1


def test_no_explicit_inputs_outputs_bad_return():
    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register()
        def f(a, b):
            return a + b

    expected = ('Variable name or Tuple of variable '
                'names are expected, got BinOp')
    assert str(err.value) == expected
