from alchemy_cat.dag.io import Input, Output


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
