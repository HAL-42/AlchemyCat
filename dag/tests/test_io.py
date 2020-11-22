from alchemy_cat.dag.io import Input, Output
import pytest
from alchemy_cat.dag.errors import PyungoError


def test_Input():
    inp = Input(name='a')
    assert inp.name == 'a'
    assert inp.value is None
    assert inp.is_kwarg is False
    assert inp.is_constant is False


def test_Input_kwarg():
    inp = Input(name='a', is_kwarg=True)
    assert inp.name == 'a'
    assert inp.value is None
    assert inp.is_kwarg is True
    assert inp.is_constant is False


def test_Input_constant():
    inp = Input(name='a', value=2)
    assert inp.name == 'a'
    assert inp.value == 2
    assert inp.is_kwarg is False
    assert inp.is_constant is True


def test_Input_constant_kwarg():
    inp = Input(name='a', value=2, is_kwarg=True)
    assert inp.name == 'a'
    assert inp.value == 2
    assert inp.is_kwarg is True
    assert inp.is_constant is True


def test_Output():
    out = Output('a')
    assert out.name == 'a'
    out.value = 2
    assert out._value == 2
    assert out.value == 2


def test_io_maps():
    for Class in [Input, Output]:
        io = Class(name='a')
        assert io.map == 'a'
        io = Class(name='a', map='b')
        assert io.map == 'b'

def test_dot_in_name():
    with pytest.raises(PyungoError) as err:
        inp = Input(name='a.b')

    assert "IO can't have '.' in name, however get name = a.b" in str(err.value)

def test_dot_in_map():
    with pytest.raises(PyungoError) as err:
        inp = Input(name='a', map='a.b')

    assert "IO can't have '.' in map, however get map = a.b" in str(err.value)
