
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.layers.python.layers import layers
import logging
import time
from tensorflow.python.ops.nn_ops import conv2d
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
import tensorflow_probability as tfp
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMCell

#from data_v2 import *
#from params import *

tfd = tfp.distributions

class BayesianLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Implementation of Bayesian LSTM Cell from
    https://gist.github.com/windweller/500ddc19d0c3cf1eb03cf73cc6b88fe3/revisions
    """
    def __init__(self, num_units,
                 prior,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 activation=tf.tanh):

        # once generated they stay the same across time-steps
        # must construct different cell for each layer
        self.prior = prior
        self.W, self.b = None, None

        self.W_mu, self.W_std = None, None
        self.b_mu, self.b_std = None, None

        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift

        super(BayesianLSTMCell, self).__init__(num_units=num_units,
                                               forget_bias=forget_bias,
                                               state_is_tuple=state_is_tuple,
                                               activation=activation)  # input_size

    # we'll see if this implementation is correct
    def get_W(self, total_arg_size, output_size, dtype):
        """
        Gets the weight parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param total_arg_size:
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellWeight"):
            if self.W is None:
                # can use its own init_scale
                self.W, self.W_mu, self.W_std = get_random_normal_variable("Matrix", self.prior,
                                                                           [total_arg_size, output_size], dtype=dtype)
        return self.W

    def get_b(self, output_size, dtype):
        """
        Gets the bias parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellBias"):
            if self.b is None:
                self.b, self.b_mu, self.b_std = get_random_normal_variable("Bias", self.prior,
                                                                           [output_size], dtype=dtype)
        return self.b

    def get_kl(self):
        """
        get the KL divergence for both the weights and the biases
        :return:
        """
        theta_kl = self.prior.get_kl_divergence((self.W_mu, self.W_std))
        theta_kl += self.prior.get_kl_divergence((self.b_mu, self.b_std))
        return theta_kl

    def _norm(self, inp, scope, dtype=dtypes.float32):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._norm_gain)
        beta_init = init_ops.constant_initializer(self._norm_shift)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
            vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def stochastic_linear(self, args, output_size, bias=True, bias_start=0.0, scope=None):
        # Local reparameterization trick
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = self.get_W(total_arg_size, output_size, dtype=dtype)
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(args, axis=1), matrix)

            if not bias:
                return res
            else:
                bias_term = self.get_b(output_size, dtype=dtype)
                return res + bias_term

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                one = constant_op.constant(1, dtype=dtypes.int32)
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)  # tf.split(state, 2, axis=1
            concat = self.stochastic_linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, axis=1)
            if self._layer_norm:
                i = self._norm(i, "input", dtype=inputs.dtype)
                j = self._norm(j, "transform", dtype=inputs.dtype)
                f = self._norm(f, "forget", dtype=inputs.dtype)
                o = self._norm(o, "output", dtype=inputs.dtype)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                              tf.nn.tanh(j))
            new_h = tf.nn.tanh(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], axis=1)
            return new_h,


class BayesianGRUCell(tf.nn.rnn_cell.GRUCell):
    """
    Implementation of Bayesian LSTM Cell from
    https://gist.github.com/windweller/500ddc19d0c3cf1eb03cf73cc6b88fe3/revisions
    """
    def __init__(self, num_units,
                 prior,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 activation=tf.tanh):

        # once generated they stay the same across time-steps
        # must construct different cell for each layer
        self.prior = prior
        self.W, self.b = None, None

        self.W_mu, self.W_std = None, None
        self.b_mu, self.b_std = None, None

        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift

        super(BayesianGRUCell, self).__init__(num_units=num_units,
                                               activation=activation)  # input_size

    # we'll see if this implementation is correct
    def get_W(self, total_arg_size, output_size, dtype):
        """
        Gets the weight parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param total_arg_size:
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellWeight"):
            if self.W is None:
                # can use its own init_scale
                self.W, self.W_mu, self.W_std = get_random_normal_variable("Matrix", self.prior,
                                                                           [total_arg_size, output_size], dtype=dtype)
        return self.W

    def get_b(self, output_size, dtype):
        """
        Gets the bias parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellBias"):
            if self.b is None:
                self.b, self.b_mu, self.b_std = get_random_normal_variable("Bias", self.prior,
                                                                           [output_size], dtype=dtype)
        return self.b

    def get_kl(self):
        """
        get the KL divergence for both the weights and the biases
        :return:
        """
        theta_kl = self.prior.get_kl_divergence((self.W_mu, self.W_std))
        theta_kl += self.prior.get_kl_divergence((self.b_mu, self.b_std))
        return theta_kl

    def _norm(self, inp, scope, dtype=dtypes.float32):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._norm_gain)
        beta_init = init_ops.constant_initializer(self._norm_shift)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
            vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def stochastic_linear(self, args, output_size, bias=True, bias_start=0.0, scope=None):
        # Local reparameterization trick
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = self.get_W(total_arg_size, output_size, dtype=dtype)
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(args, axis=1), matrix)

            if not bias:
                return res
            else:
                bias_term = self.get_b(output_size, dtype=dtype)
                return res + bias_term



    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):


            gate_inputs = self.stochastic_linear([inputs, state], 2* self._num_units, True)

            value = math_ops.sigmoid(gate_inputs)
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

            if self._layer_norm:
                    r = self._norm(r, "reset", dtype=inputs.dtype)
                    u = self._norm(u, "update", dtype=inputs.dtype)

            r_state = r * state

            candidate = math_ops.matmul(
                array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
            candidate = nn_ops.bias_add(candidate, self._candidate_bias)

            c = self._activation(candidate)
            new_h = u * state + (1 - u) * c
            return new_h, new_h



class BayesianRNNrelu(tf.nn.rnn_cell.BasicRNNCell):
    """
    Implementation of Bayesian LSTM Cell from
    https://gist.github.com/windweller/500ddc19d0c3cf1eb03cf73cc6b88fe3/revisions
    """
    def __init__(self, num_units,
                 prior,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 activation=tf.tanh):

        # once generated they stay the same across time-steps
        # must construct different cell for each layer
        self.prior = prior
        self.W, self.b = None, None

        self.W_mu, self.W_std = None, None
        self.b_mu, self.b_std = None, None

        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift

        super(BayesianRNNrelu, self).__init__(num_units=num_units,
                                               activation=activation)  # input_size

    # we'll see if this implementation is correct
    def get_W(self, total_arg_size, output_size, dtype):
        """
        Gets the weight parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param total_arg_size:
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellWeight"):
            if self.W is None:
                # can use its own init_scale
                self.W, self.W_mu, self.W_std = get_random_normal_variable("Matrix", self.prior,
                                                                           [total_arg_size, output_size], dtype=dtype)
        return self.W

    def get_b(self, output_size, dtype):
        """
        Gets the bias parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellBias"):
            if self.b is None:
                self.b, self.b_mu, self.b_std = get_random_normal_variable("Bias", self.prior,
                                                                           [output_size], dtype=dtype)
        return self.b

    def get_kl(self):
        """
        get the KL divergence for both the weights and the biases
        :return:
        """
        theta_kl = self.prior.get_kl_divergence((self.W_mu, self.W_std))
        theta_kl += self.prior.get_kl_divergence((self.b_mu, self.b_std))
        return theta_kl

    def _norm(self, inp, scope, dtype=dtypes.float32):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._norm_gain)
        beta_init = init_ops.constant_initializer(self._norm_shift)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
            vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def stochastic_linear(self, args, output_size, bias=True, bias_start=0.0, scope=None):
        # Local reparameterization trick
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = self.get_W(total_arg_size, output_size, dtype=dtype)
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(args, axis=1), matrix)

            if not bias:
                return res
            else:
                bias_term = self.get_b(output_size, dtype=dtype)
                return res + bias_term

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):

            #######################################################
            concat = self.stochastic_linear([inputs, state], self._num_units, True) # gate inputs
            #######################################################


            #######################################################
            if self._layer_norm:   #this is different
                gate_inputs = self._norm(concat, "input", dtype=inputs.dtype)
            #######################################################

            output = tf.nn.relu(gate_inputs)

            return output, output





class BayesianRNNtanh(tf.nn.rnn_cell.BasicRNNCell):
    """
    Implementation of Bayesian LSTM Cell from
    https://gist.github.com/windweller/500ddc19d0c3cf1eb03cf73cc6b88fe3/revisions
    """
    def __init__(self, num_units,
                 prior,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 activation=tf.tanh):

        # once generated they stay the same across time-steps
        # must construct different cell for each layer
        self.prior = prior
        self.W, self.b = None, None

        self.W_mu, self.W_std = None, None
        self.b_mu, self.b_std = None, None

        self._layer_norm = layer_norm
        self._norm_gain = norm_gain
        self._norm_shift = norm_shift

        super(BayesianRNNtanh, self).__init__(num_units=num_units,
                                               activation=activation)  # input_size

    # we'll see if this implementation is correct
    def get_W(self, total_arg_size, output_size, dtype):
        """
        Gets the weight parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param total_arg_size:
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellWeight"):
            if self.W is None:
                # can use its own init_scale
                self.W, self.W_mu, self.W_std = get_random_normal_variable("Matrix", self.prior,
                                                                           [total_arg_size, output_size], dtype=dtype)
        return self.W

    def get_b(self, output_size, dtype):
        """
        Gets the bias parameter
        On each call, a new parameter will be sampled.
        At test time, it returns the MAP
        :param output_size:
        :param dtype:
        :return:
        """
        with tf.variable_scope("CellBias"):
            if self.b is None:
                self.b, self.b_mu, self.b_std = get_random_normal_variable("Bias", self.prior,
                                                                           [output_size], dtype=dtype)
        return self.b

    def get_kl(self):
        """
        get the KL divergence for both the weights and the biases
        :return:
        """
        theta_kl = self.prior.get_kl_divergence((self.W_mu, self.W_std))
        theta_kl += self.prior.get_kl_divergence((self.b_mu, self.b_std))
        return theta_kl

    def _norm(self, inp, scope, dtype=dtypes.float32):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._norm_gain)
        beta_init = init_ops.constant_initializer(self._norm_shift)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init, dtype=dtype)
            vs.get_variable("beta", shape=shape, initializer=beta_init, dtype=dtype)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def stochastic_linear(self, args, output_size, bias=True, bias_start=0.0, scope=None):
        # Local reparameterization trick
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = self.get_W(total_arg_size, output_size, dtype=dtype)
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(args, axis=1), matrix)

            if not bias:
                return res
            else:
                bias_term = self.get_b(output_size, dtype=dtype)
                return res + bias_term

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"


            #######################################################
            concat = self.stochastic_linear([inputs, state], self._num_units, True) # gate inputs
            #######################################################


            #######################################################
            if self._layer_norm:   #this is different
                gate_inputs = self._norm(concat, "input", dtype=inputs.dtype)
            #######################################################

            output = tf.nn.tanh(gate_inputs)

            return output, output
