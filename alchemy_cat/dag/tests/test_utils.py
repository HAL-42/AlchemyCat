import pytest

from alchemy_cat.dag.errors import PyungoError
from alchemy_cat.dag.utils import get_function_return_names


def test_get_function_return_names_simple():

    def a():
        b = 2
        return b

    assert get_function_return_names(a) == ['b']


def test_get_function_return_names_no_return():

    def a():
        pass

    with pytest.raises(PyungoError) as err:
        get_function_return_names(a)

    assert str(err.value) == 'No return statement found in a'


def test_get_function_return_names_tuple():

    def a():
        b = 2
        c = 3
        return b, c

    assert get_function_return_names(a) == ['b', 'c']


def test_get_function_return_names_not_valid():

    def a():
        b = 2
        c = 3
        return b / c

    with pytest.raises(PyungoError) as err:
        get_function_return_names(a)

    msg = 'Variable name or Tuple of variable names are expected, got BinOp'
    assert str(err.value) == msg
