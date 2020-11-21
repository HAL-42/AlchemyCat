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

    # make sure it is independent
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5


def test_call():
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

    for _ in range(2):
        res = graph(a=2, b=3)
        assert res == -1.5
        assert graph.data['e'] == -1.5

    with pytest.raises(PyungoError, match="Graph only receive keyword args which will be "
                                          "recognized as input name and value."):
        graph(2, 3)


def test_inputs_args_equivalent():
    graph = Graph()

    @graph.register(args=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(args=['d', 'a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(args=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 10.

    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5

    # make sure it is independent
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5


def test_inputs_args_blend():
    graph = Graph()

    @graph.register(inputs=['a'], args=['b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    @graph.register(inputs=['d'], args=['a'], outputs=['e'])
    def f_my_function3(d, a):
        return d - a

    @graph.register(args=['c'], outputs=['d'])
    def f_my_function2(c):
        return c / 10.

    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5

    # make sure it is independent
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5


def test_simple_constant_inputs():
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

    res = graph.calculate(data={'a': 2})
    assert res == -1.5
    assert graph.data['e'] == -1.5

    # make sure it is independent
    res = graph.calculate(data={'a': 2})
    assert res == -1.5
    assert graph.data['e'] == -1.5


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


def test_complex_constant_inputs():
    """Test complex constant inputs.
        * Test {k:v, k:v, ...} constant
        * Test Input(name, value=*) constant
        * Test kwargs constant
    """
    graph = Graph()

    # f1 = -1
    @graph.register(inputs={'inp_1_1': 2, 'inp_1_2': 3}, kwargs={'inp_1_3': 6}, outputs=['f1'])
    def f_my_function1(inp_1_1, inp_1_2=2, inp_1_3=3):
        return inp_1_1 + inp_1_2 - inp_1_3

    # f2 = (2, -5)
    @graph.register(args=['f1', Input('i_2_2', value=-1), {'i_2_3_1': 1}, {'i_2_3_2': -2}],
                    kwargs=[Input('inp_2_4', value=3)], outputs=['f2'])
    def f_my_function2(inp_2_1=4, inp_2_2=5, *inp_2_3, **inp_2_4):
        return inp_2_1 * inp_2_2 + inp_2_3[0], inp_2_3[1] - list(inp_2_4.values())[0]

    # f3 = -33
    @graph.register(inputs=['f1', 'f2', 'inp_3_3'], outputs=['f3'])
    def f_my_function3(inp_3_1, inp_3_2, inp_3_3=4):
        return (inp_3_1 - inp_3_2[0] * inp_3_2[1]) * inp_3_3

    for _ in range(2):
        res = graph.calculate(data={'inp_3_3': 3})
        assert res == 27
        assert graph.data['f3'] == res
        assert graph.data['f1'] == -1
        assert graph.data['f2'] == (2, -5)


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

    for _ in range(2):
        data={'a': np.ones((2, 2)) * 2., 'b': np.ones((2, 2)) * 3.}
        res = graph.calculate(data)
        assert (res == np.ones((2, 2)) * -1.5).all()
        assert (graph.data['e'] == np.ones((2, 2)) * -1.5).all()
        assert id(graph.ordered_nodes[0]._outputs[0].value) == id(graph.ordered_nodes[1]._inputs[0].value)
        assert id(graph.ordered_nodes[1]._outputs[0].value) == id(graph.ordered_nodes[2]._inputs[0].value)
        assert id(data['a']) == id(graph.ordered_nodes[0]._inputs[0].value)
        assert id(data['b']) == id(graph.ordered_nodes[0]._inputs[1].value)


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

    for _ in range(2):
        data={'a': np.ones((2, 2)) * 2., 'b': np.ones((2, 2)) * 3.}
        res = graph.calculate(data)
        assert (res == np.ones((2, 2)) * -1.5).all()
        assert (graph.data['e'] == np.ones((2, 2)) * -1.5).all()
        assert id(graph.ordered_nodes[0]._outputs[0].value) != id(graph.ordered_nodes[1]._inputs[0].value)
        assert id(graph.ordered_nodes[1]._outputs[0].value) != id(graph.ordered_nodes[2]._inputs[0].value)
        assert id(data['a']) != id(graph.ordered_nodes[0]._inputs[0].value)
        assert id(data['b']) != id(graph.ordered_nodes[0]._inputs[1].value)


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

    res = graph.calculate(data={'a': np.ones((2, 2)) * 2., 'b': np.ones((2, 2)) * 3.})
    assert (res == np.ones((2, 2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2, 2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) == id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) != id(graph.ordered_nodes[2]._inputs[0].value)

    res = graph.calculate(data={'a': np.ones((2, 2)) * 2., 'b': np.ones((2, 2)) * 3.})
    assert (res == np.ones((2, 2)) * -1.5).all()
    assert (graph.data['e'] == np.ones((2, 2)) * -1.5).all()
    assert id(graph.ordered_nodes[0]._outputs[0].value) == id(graph.ordered_nodes[1]._inputs[0].value)
    assert id(graph.ordered_nodes[1]._outputs[0].value) != id(graph.ordered_nodes[2]._inputs[0].value)


def test_functor():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'], init={})
    class f_my_function(object):
        factor = 1

        def __init__(self):
            self.dummy = 100

        def __call__(self, a1, a2):
            return a1 + a2 + self.factor

    @graph.register(inputs=['d', 'a'], outputs=['e'], init={'add_or_sub': 'add', 'factor': 2})
    class f_my_function2(object):
        def __init__(self, add_or_sub='add', factor=1):
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
        def __init__(self, add_or_sub='add', factor=1):
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


def test_simple_parallel():
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

    assert "Node Node(<f_my_function2>, ['c'], ['c']) have repeated output names: ['c']" in str(err.value)


def test_missing_input():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6})

    assert "The following inputs are needed: ['b']" in str(err.value)


def test_missing_kwargs():
    graph = Graph()

    @graph.register(inputs=['a'], kwargs=['b'],outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6})

    assert "The following inputs are needed: ['b']" in str(err.value)


def test_missing_input_both_nec_opt():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b=2):
        return a + b

    @graph.register(kwargs=['a', 'b'], outputs=['e'])
    def f_my_function3(a, b):
        return a - b

    @graph.register(inputs=['c', 'e'], outputs=['f'])
    def f_my_function2(c, e):
        return c + e / 10.

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


def test_inputs_not_used_with_constant():
    graph = Graph()

    @graph.register(inputs=[{'a': 1}, 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6, 'b': 4})

    assert "The following inputs are not used by the model: ['a']" in str(err.value)


def test_opt_inputs_wont_cause_redundant_input():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b=2):
        return a + b

    res = graph.calculate(data={'a': 6})
    assert res == 8


def test_inputs_collision():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        graph.calculate(data={'a': 6, 'b': 4, 'c': 7})

    assert "The following inputs are already used in the model: ['c']" in str(err.value)


def test_self_dependence():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], outputs=['c'])
    def f_my_function(a, b):
        return a + b

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=['c', 'd', 'e', {'f': 1}], outputs=['d', 'e', 'f'])
        def f_my_function2(c, d, e):
            return c, d, e

    assert "Node Node(<f_my_function2>, ['c', 'd', 'e', 'f'], ['d', 'e', 'f']) have self dependence caused " \
           "by the following inputs: ['d', 'e']" in str(err.value)


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


def test_Input_type_input():
    graph = Graph()

    @graph.register(
        inputs=[Input(name='a'), Input(name='inp_1_1', map='b')],
        outputs=['c']
    )
    def f_my_function(a, b):
        return a + b

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert res == 5


def test_input_type_tuple():
    graph = Graph()

    @graph.register(
        inputs=[('inp_1', 'a'), ('b', 'b')],
        outputs=['c']
    )
    def f_my_function(a, b):
        return a + b

    res = graph.calculate(data={'a': 2, 'b': 3})

    assert res == 5

def test_wrong_input_type():
    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=['a', {'b'}], outputs=['c'])
        def f_my_function(a, b):
            return a + b

    assert "inputs need to be of type tuple, Input, str or dict" in str(err.value)


def test_input_tuple_too_long():
    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=[('a', 'input', 'too_long'), 'b'], outputs=['c'])
        def f_my_function(a, b):
            return a + b

    assert "Tuple input should like (name, map). However, get ('a', 'input', 'too_long')" in str(err.value)


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


def test_not_str_name():
    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=[(23, 'a')], outputs=['c'])
        def f_my_function(a, b):
            return a + b

    assert "IO name must be str, however get name = 23 with type <class 'int'>" in str(err.value)


def test_not_str_map():
    graph = Graph()

    with pytest.raises(PyungoError) as err:
        @graph.register(inputs=[Input('a', map=23)], outputs=['c'])
        def f_my_function(a, b):
            return a + b

    assert "IO map must be str, however get map = 23 with type <class 'int'>" in str(err.value)


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


def test_build_with_map_feed_with_name():
    graph = Graph()

    @graph.register(inputs=[('foo', 'a')], kwargs=[('inp1_2', 'b')],outputs=['c'])
    def f_my_function1(inp1_1, inp1_2):
        return inp1_1 + inp1_2

    @graph.register(args=[Input(name='foo', map='d')], kwargs=[Input(name='inp3_2', map='a')], outputs=['e'])
    def f_my_function3(inp3_1, inp3_2):
        return inp3_1 - inp3_2

    @graph.register(inputs=[('foo', 'c')], outputs=['d'])
    def f_my_function2(inp2_1):
        return inp2_1 / 10.

    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5

    # make sure it is independent
    res = graph.calculate(data={'a': 2, 'b': 3})
    assert res == -1.5
    assert graph.data['e'] == -1.5


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


def test_find_default_by_name_not_map():
    graph = Graph()

    @graph.register(inputs=['a', ('inp2', 'b')], kwargs=[('inp3', 'c')], outputs=['d'])
    def f(inp1, inp2=2, inp3=3):
        return inp1 + inp2 + inp3

    res = graph.calculate(data={'a': 1})

    assert res == 6
    assert graph.data['d'] == 6


def test_optional_inputs_without_feed():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], kwargs=['c'], outputs=['d'])
    def f(a, b=2, c=3):
        return a + b + c

    res = graph.calculate(data={'a': 1})

    assert res == 6
    assert graph.data['d'] == 6


def test_optional_inputs_feed_by_input():
    graph = Graph()

    @graph.register(inputs=['a', 'b'], kwargs=['c'], outputs=['d'])
    def f(a, b=2, c=3):
        return a + b + c

    res = graph.calculate(data={'a': 1, 'b': -1, 'c': -2})

    assert res == -2
    assert graph.data['d'] == -2


def test_optional_inputs_feed_by_output():
    graph = Graph()

    @graph.register(inputs=['a'], kwargs=['b'], outputs=['c'])
    def f(a, b):
        return a + b

    @graph.register(inputs=['a'], kwargs=['b'], outputs=['d'])
    def f2(a, b):
        return a - b

    @graph.register(inputs=['c'], kwargs=[Input(map='d', name='inp2')], outputs=['e'])
    def f1(inp1=0, inp2=0):
        return inp1 + inp2

    res = graph.calculate(data={'a': 1, 'b': 3})

    assert res == 2
    assert graph.data['e'] == res
    assert graph.data['c'] == 4
    assert graph.data['d'] == -2


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

##

