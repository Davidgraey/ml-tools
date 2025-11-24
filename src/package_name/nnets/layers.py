import numpy as np
import activation_functions as func
import copy
from abc import ABC, ABCMeta

# https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/
# MISH - A self-regularized Non-monotonic Activation Function

# TODO: expand acitvation functions
# TODO: add GELU
# TODO: add swish
# TODO: add MISH - x * tanh(softplus(x))
# TODO: implement in jax
# import jax.numpy as jnp

# TODO: cache with LRU cache?  could be useful with rounding or processing
#  int16/8 values rather than floats
# from functools import lru_cache



activation_weight_scaler = {'linear': 0.5,
                            'relu': 0.5,
                            'relu_leaky': 0.5,
                            'sigmoid': 1.0,
                            'tanh': 1.0,
                            'softmax': 1.0} # 2


class Layer(ABC):
    def __init__(self, ni, no, activation_type, is_output=False):
        '''**********ARGUMENTS**********
        :param ni: number of input units
        :param no: number of output units
        :param activation_type: string identifying activation type, 'linear', 'sigmoid', 'tanh', etc.
        :param is_output: boolean flag designating if this is an output layer or hidden layer
        '''
        self.activation = activation_type
        self.is_output = is_output
        self.weights = np.random.normal(loc=0.0,
                                        scale=activation_weight_scaler[activation_type],
                                        size=(ni + 1, no))  # adding one to number of inputs (ni) for a bias weight
        self.shape = self.weights.shape

        # these values will be rewritten or updated on each pass
        self.output = 0.0
        self.input = 0.0
        self.gradient = 0.0


    def forward(self, incoming_x, forced_activation = False):
        '''**********ARGUMENTS**********
        :param incoming_x: input data that is already standardized, if called for
        **********RETURNS**********
        :return: product of forward pass of incoming values
        '''
        self.input = copy.copy(incoming_x)
        # bias units are self.weights[0:1, :]
        if forced_activation != False:
            outs = func.activation_dictionary[forced_activation](incoming_x @ self.weights[1:, :] + self.weights[0:1,
                                                                                                    :])
        else:
            outs = func.activation_dictionary[self.activation](incoming_x @ self.weights[1:, :] + self.weights[0:1, :])
            self.output = outs
        return outs


    def build_gradient(self, incoming_delta):
        '''**********ARGUMENTS**********
        :param incoming_delta: delta from previous step / original error
        **********RETURNS**********
        :return: passes gradient back for backpropigation
        '''
        #print(f'delta is {incoming_delta.shape}, outs are {self.output.shape}, ins are {self.input.shape}')
        incoming_delta *= func.derivative_dictionary[self.activation](self.output)
        this_delta = self.input.T @ incoming_delta
        bias_delta = np.sum(incoming_delta, 0)
        #this_delta = np.expand_dims(np.squeeze(this_delta).reshape(-1), 1)
        #self.gradient = np.vstack((bias_delta, this_delta))
        self.gradient = np.vstack((bias_delta, this_delta))
        #print(f'in build_gradient grad is {self.gradient.shape}')
        return self.gradient


    def back_pass_gradient(self, incoming_gradient):
        '''**********ARGUMENTS**********
        :param incoming_gradient: this layer's gradient, created by self.build_gradient()
        **********RETURNS**********
        :return: returns this layer's gradient contribution
        '''
        grad_contribution = incoming_gradient @ self.weights[1:, ...].T
        #print(f'passed grad back through {self}')
        return grad_contribution


    def update_weight(self, value):
        '''**********ARGUMENTS**********
        :param value: variable to update this layer's weights by with - this will already have Learning rate,
        depreication or momentum / other calculations addressed in the upper level.
        '''
        self.weights += value


    def purge(self):
        '''
        resets values tracked during training to zero
        '''
        self.input = 0.0
        self.output = 0.0
        self.gradient = 0.0

    def __str__(self):
        return f'Layer of {self.activation}, shaped {self.shape}'

    def __repr__(self):
        return f'Layer of {self.activation}, shaped {self.shape}'