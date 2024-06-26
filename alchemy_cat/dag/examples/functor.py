import numpy as np

from alchemy_cat.dag.core import Graph


def f_my_function(a1, a2):
        return a1 + a2 + 1


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


if __name__ == '__main__':
    graph = Graph()

    def register_graph(graph):
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

        graph.add_node(f_my_function2, inputs=['d', 'a'], outputs=['e'], init={'add_or_sub': 'add', 'factor': 2})
        graph.add_node(f_my_function, inputs=['a', 'b'], outputs=['c'])

    register_graph(graph)

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