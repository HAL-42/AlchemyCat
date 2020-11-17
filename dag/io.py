""" Input / Output module

Inputs / Outputs objects used to represent functions inputs / outputs
"""


class _IO(object):
    """ IO Base class for inputs and outputs

    Args:
        name (str): The variable name of the input / output
        map (str): Optional mapping name. This is the name referenced
            in the data inputs / outputs
        meta (dict): Not used yet
        contract (str): Optional contract rule used by pycontracts
    """

    def __init__(self, name, map=None, meta=None, contract=None):
        self._name = name
        self._meta = meta if meta is not None else {}
        self._value = None
        self._contract = None
        self._map = map if map else self._name
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
        name (str): The variable name of the input / output
        map (str): Optional mapping name. This is the name referenced
            in the data inputs / outputs. For Data object, they implement
            the Node's with their map.
        meta (dict): Not used yet
        contract (str): Optional contract rule used by pycontracts
    """

    def __init__(self, name, map=None, meta=None, contract=None):
        super(Input, self).__init__(name, map, meta, contract)
        self.is_arg = False
        self.is_kwarg = False
        self.is_constant = False

    def __repr__(self):
        return '<{} value={} is_arg: {} is_kwarg: {}>'.format(
            self._name, self.value, self.is_arg, self.is_kwarg
        )

    @classmethod
    def constant(cls, name, value, meta=None, is_arg=False, is_kwarg=False):
        """ Alternate constructor for inputs that are constant

        Args:
            name (str): The variable name of the input / output
            value: The defined constant value, can be anything
            meta (dict): Not used yet
            is_arg (bool): Whether the Input is constant arg
            is_kwarg (bool): Whether the Input is constant kwarg
        """
        me = cls(name, meta=meta)
        me.value = value
        me.is_constant = True
        me.is_arg, me.is_kwarg = is_arg, is_kwarg
        return me

    @classmethod
    def arg(cls, name, meta=None):
        """ Alternate constructor for args

        Args:
            name (str): The variable name of the input / output
            meta (dict): Not used yet
        """
        me = cls(name, meta=meta)
        me.is_arg = True
        return me

    @classmethod
    def kwarg(cls, name, meta=None):
        """ Alternate constructor for kwargs

        Args:
            name (str): The variable name of the input / output
            meta (dict): Not used yet
        """
        me = cls(name, meta=meta)
        me.is_kwarg = True
        return me


class Output(_IO):
    """ Placeholder of a function's output"""
    def __repr__(self):
        return '<{} value={}>'.format(
            self._name, self.value
        )


def get_if_exists(provided, existing):
    """Process the inputs/outputs of an new node.
    The inputs/outputs of node will be replaced by graph's _inputs/_outputs if they share
    the same name. So different nodes can use the same Input, and multi output can be gotten
    from the graph.

    Args:
        provided (list): Inputs/Outputs of Node
        existing (dict): _inputs/_outputs of the Graph

    Returns:
        res (list): The replaced inputs/outputs
    """
    if not existing or not provided:
        return provided
    res = []
    for p in provided:
        is_io = False
        if isinstance(p, str):
            name = p
        elif isinstance(p, Input) or isinstance(p, Output):
            name = p.name
            is_io = True
        else:
            res.append(p)
            continue
        exist = existing.get(name)
        if exist:
            if is_io:
                msg = ('You cannot use Input / Output in a '
                       'Node if already defined')
                raise TypeError(msg)
            res.append(exist)
        else:
            res.append(p)
    return res
