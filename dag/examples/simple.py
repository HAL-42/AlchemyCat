import numpy as np

from alchemy_cat.dag.core import Graph

graph = Graph(pool_size=2, verbosity=2, slim=False)

@graph.register()
def f_my_function(a, b):
    c = a + b
    return c
@graph.register(inputs=['d', 'a'], outputs=['e'])
def f_my_function3(d, a):
    e = d - a
    return e
@graph.register(outputs=['d'])
def f_my_function2(c):
    return c / 10.

if __name__ == '__main__':
    res = graph.calculate(data={'a': np.random.randn(2000, 2000), 'b': np.random.randn(2000, 2000)})
    print(res)
