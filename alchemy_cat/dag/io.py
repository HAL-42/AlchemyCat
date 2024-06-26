""" Input / Output module

Inputs / Outputs objects used to represent functions inputs / outputs
"""
from alchemy_cat.dag.errors import PyungoError


class _IO(object):
    """ IO Base class for inputs and outputs

    Args:
        name (str): The variable name of the input / output. Used when feeding input to node's func or find input's
            default value.
        map (str): Optional mapping name. Used when building graph.
        meta (dict): Not used yet.
        contract (str): Optional contract rule used by pycontracts.
    """

    def __init__(self, name, map=None, meta=None, contract=None):
        if not isinstance(name, str):
            raise PyungoError(f"IO name must be str, however get name = {name} with type {type(name)}")
        if '.' in name:
            raise PyungoError(f"IO can't have '.' in name, however get name = {name}")
        self._name = name

        self._meta = meta if meta is not None else {}
        self._value = None
        self._contract = None

        if map is not None:
            if not isinstance(map, str):
                raise PyungoError(f"IO map must be str, however get map = {map} with type {type(map)}")
            if '.' in map:
                raise PyungoError(f"IO can't have '.' in map, however get map = {map}")
        self._map = map if map is not None else self._name

        if contract:
            try:
                from contracts.main import parse_contract_string
            except ImportError:
                raise ImportError('pycontracts is needed to use contracts')
            self._contract = parse_contract_string(contract)

    @property
    def name(self):
        return self._name

    @property
    def map(self):
        return self._map

    @property
    def contract(self):
        return self._contract

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, x):
        """ When setting a value, we check the contract when applicable """
        if self._contract:
            self._contract.check(x)
        self._value = x


class Input(_IO):
    """ Placeholder a function's input

    Args:
        name (str): The variable name of the input / output. Used when feeding input to node's func or find input's
            default value.
        map (str): Optional mapping name. Used when building graph.
        is_kwarg (bool): Is kwarg input.
        value (Any): If not None, inp.value = value and is_constant = True.
        meta (dict): Not used yet.
        contract (str): Optional contract rule used by pycontracts.
    """

    def __init__(self, name, map=None, is_kwarg=False, value=None, meta=None, contract=None):
        super(Input, self).__init__(name, map, meta, contract)
        self.is_kwarg = is_kwarg
        self.is_constant = True if value is not None else False
        self.value = value

    def __repr__(self):
        return '<{} value={} is_kwarg: {} is_constant>'.format(
            self._name, self.value, self.is_kwarg, self.is_constant
        )


class Output(_IO):
    """ Placeholder of a function's output"""
    def __repr__(self):
        return '<{} value={}>'.format(
            self._name, self.value
        )
