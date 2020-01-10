""" class abstracting data passed as inputs and saved as outputs """
from copy import deepcopy

from alchemy_cat.dag.errors import PyungoError
from alchemy_cat.dag.io import Output


class Data:
    def __init__(self, inputs):
        self._inputs = deepcopy(inputs)
        self._outputs= {}

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __getitem__(self, key):
        try:
            return self._inputs[key]
        except KeyError:
            return self._outputs[key].value

    def __setitem__(self, map: str, output: Output):
        if map != output.map:
            raise PyungoError(f"Key {map} should be equal to Output's map {output.map}")
        self._outputs[map] = output

    def check_inputs(self, sim_inputs, sim_outputs, sim_kwargs):
        """ make sure data inputs provided are good enough """
        # inputs in data can't have the same name of any the output
        data_inputs = set(self.inputs.keys())
        diff = data_inputs - (data_inputs - set(sim_outputs))
        if diff:
            msg = 'The following inputs are already used in the model: {}'
            raise PyungoError(msg.format(list(diff)))
        # inputs in data should able to provide inputs needed for calculate
        inputs_to_provide = set(sim_inputs) - set(sim_outputs)
        diff = inputs_to_provide - data_inputs
        if diff:
            msg = 'The following inputs are needed: {}'.format(list(diff))
            raise PyungoError(msg)
        # All inputs should be used in the calculation
        diff = data_inputs - inputs_to_provide - set(sim_kwargs)
        if diff:
            msg = 'The following inputs are not used by the model: {}'
            raise PyungoError(msg.format(list(diff)))
        # kwarg input can't be provided twice in both data inputs and sim_outputs
        diff = set(sim_kwargs) & set(sim_outputs) & data_inputs
        if diff:
            msg = 'The following kwarg inputs value will be provided twice by both data inputs and sim_outputs: {}'
            raise PyungoError(msg.format(list(diff)))

