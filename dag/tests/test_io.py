from alchemy_cat.dag.io import Input, Output, get_if_exists


def test_Input():
    inp = Input(name='a')
    assert inp.name == 'a'
    assert inp.value is None
    assert inp.is_arg is False
    assert inp.is_kwarg is False


def test_Input_arg():
    inp = Input.arg(name='a')
    assert inp.name == 'a'
    assert inp.value is None
    assert inp.is_arg is True
    assert inp.is_kwarg is False


def test_Input_kwarg():
    inp = Input.kwarg(name='a')
    assert inp.name == 'a'
    assert inp.value is None
    assert inp.is_arg is False
    assert inp.is_kwarg is True


def test_Input_constant():
    inp = Input.constant(name='a', value=2)
    assert inp.name == 'a'
    assert inp.value == 2
    assert inp.is_arg is False
    assert inp.is_kwarg is False
    assert inp.is_constant is True


def test_Output():
    out = Output('a')
    assert out.name == 'a'
    out.value = 2
    assert out._value == 2
    assert out.value == 2


def test_get_if_exists_do_not_exists():
    res = get_if_exists([1, 2, 3], {})
    assert res == [1, 2, 3]


def test_get_if_exists_ok():
    inputs = [Input('a'), Input('b'), Input('c')]
    existing = {i.name: i for i in inputs}
    res = get_if_exists(['a', 'b', 'c', {'d': 3}], existing)
    for i, j in zip(inputs, res):  # skip the dict on purpose
        assert i == j


def test_io_maps():
    for Class in [Input, Output]:
        io = Class(name='a')
        assert io.map == 'a'
        io = Class(name='a', map='b')
        assert io.map == 'b'
