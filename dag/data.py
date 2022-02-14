""" class abstracting data passed as inputs and saved as outputs """
from copy import deepcopy

from alchemy_cat.dag.errors import PyungoError
from alchemy_cat.dag.io import Output


class Data:
    def __init__(self, inputs, slim=False):
        self._slim = slim

        if self._slim:
            self._inputs = inputs
        else:
            self._inputs = deepcopy(inputs)

        self._outputs = {}

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __getitem__(self, map):
        try:
            return self._inputs[map]
        except KeyError:
            return self._outputs[map]

    def __setitem__(self, map, output_value):
        self._outputs[map] = output_value

    def __contains__(self, map):
        return (map in self._inputs) or (map in self._outputs)

    def check_inputs(self, dag_nec_input_maps: set, dag_opt_input_maps: set, dag_output_names: set):
        """ Check input is legal. """
        # inputs in data can't have the same name of any the output
        data_input_maps = set(self.inputs.keys())
        dag_nec_input_maps = dag_nec_input_maps
        dag_opt_input_maps = dag_opt_input_maps
        dag_output_names = dag_output_names

        diff = data_input_maps & dag_output_names
        if diff:
            msg = 'The following inputs are already used in the model: {}'
            raise PyungoError(msg.format(sorted(list(diff))))
        # inputs in data should be able to provide inputs needed for calculate
        diff = dag_nec_input_maps - dag_output_names - data_input_maps
        if diff:
            msg = 'The following inputs are needed: {}'.format(sorted(list(diff)))
            raise PyungoError(msg)
        # All inputs should be used in the calculation
        diff = data_input_maps - dag_nec_input_maps - dag_opt_input_maps
        if diff:
            msg = 'The following inputs are not used by the model: {}'
            raise PyungoError(msg.format(sorted(list(diff))))
