import math
import keras
import tensorflow as tf
from tensorflow import distributions, matrix_band_part, igamma, lgamma
from keras import backend as K

def cons_greater_zero(value):
	epsilon = 1e-9
	return epsilon + K.pow(value, 2)

class SPP(keras.layers.Layer):
	"""
	Parametric softplus activation layer.
	"""

	def __init__(self, alpha, **kwargs):
		super(SPP, self).__init__(**kwargs)
		self.__name__ = 'SPP'
		self.alpha = alpha

	def build(self, input_shape):
		super(SPP, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return K.softplus(inputs) - self.alpha

	def compute_output_shape(self, input_shape):
		return input_shape

class SPPT(keras.layers.Layer):
	"""
	Trainable Parametric softplus activation layer.
	"""

	def __init__(self, **kwargs):
		super(SPPT, self).__init__(**kwargs)
		self.__name__ = 'SPP'

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=0, maxval=1),
									 trainable=True)

		super(SPPT, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return K.softplus(inputs) - self.alpha

	def compute_output_shape(self, input_shape):
		return input_shape


def parametric_softplus(spp_alpha):
	"""
	Compute parametric softplus function with given alpha.
	:param spp_alpha: alpha parameter for softplus function.
	:return: parametric softplus activation value.
	"""

	def spp(x):
		return K.log(1 + K.exp(x)) - spp_alpha

	return spp


class MPELU(keras.layers.Layer):
	def __init__(self, channel_wise=True, **kwargs):
		super(MPELU, self).__init__(**kwargs)
		self.channel_wise = channel_wise

	def build(self, input_shape):
		shape = [1]

		if self.channel_wise:
			shape = [int(input_shape[-1])]  # Number of channels

		self.alpha = self.add_weight(name='alpha', shape=shape, dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
									 trainable=True)
		self.beta = self.add_weight(name='beta', shape=shape, dtype=K.floatx(),
									initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1),
									trainable=True)

		# Finish buildidng
		super(MPELU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		positive = keras.activations.relu(inputs)
		negative = self.alpha * (K.exp(-keras.activations.relu(-inputs) * cons_greater_zero(self.beta)) - 1)

		return positive + negative

	def compute_output_shape(self, input_shape):
		return input_shape


class RTReLU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(RTReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = [int(input_shape[-1])]  # Number of channels

		self.a = self.add_weight(name='a', shape=shape, dtype=K.floatx(),
								 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
								 trainable=False)

		# Finish building
		super(RTReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return keras.activations.relu(inputs + self.a)

	def compute_output_shape(self, input_shape):
		return input_shape


class RTPReLU(keras.layers.PReLU):
	def __init__(self, **kwargs):
		super(RTPReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		shape = [int(input_shape[-1])]  # Number of channels

		self.a = self.add_weight(name='a', shape=shape, dtype=K.floatx(),
								 initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
								 trainable=False)

		# Call PReLU build method
		super(RTPReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = keras.activations.relu(inputs + self.a)
		neg = -self.alpha * keras.activations.relu(-(inputs * self.a))

		return pos + neg


class PairedReLU(keras.layers.Layer):
	def __init__(self, scale=0.5, **kwargs):
		super(PairedReLU, self).__init__(**kwargs)
		self.scale = scale

	def build(self, input_shape):
		self.theta = self.add_weight(name='theta', shape=[1], dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
									 trainable=True)
		self.theta_p = self.add_weight(name='theta_p', shape=[1], dtype=K.floatx(),
									   initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
									   trainable=True)

		# Finish building
		super(PairedReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return K.concatenate(
			(keras.activations.relu(self.scale * inputs - self.theta), keras.activations.relu(-self.scale * inputs - self.theta_p)),
			axis=len(inputs.get_shape()) - 1)

	def compute_output_shape(self, input_shape):
		shape = list(input_shape)
		shape[-1]  = shape[-1] * 2
		shape = tuple(shape)
		return shape


class EReLU(keras.layers.Layer):
	def __init__(self, alpha=0.5, **kwargs):
		super(EReLU, self).__init__(**kwargs)
		self.alpha = alpha

	def build(self, input_shape):
		# shape = input_shape[1:]

		# self.k = self.add_weight(name='k', shape=shape, dtype=K.floatx(),
		# 						 initializer=keras.initializers.RandomUniform(minval=1 - self.alpha, maxval=1 + self.alpha), trainable=False)

		# Finish building
		super(EReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# Generate random uniform tensor between [1-alpha, 1+alpha] for training and ones tensor for test (ReLU)
		k = K.in_train_phase(K.random_uniform(inputs.shape[1:], 1 - self.alpha, 1 + self.alpha), K.ones(inputs.shape[1:]))

		return keras.activations.relu(inputs * k)

	def compute_output_shape(self, input_shape):
		return input_shape


class EPReLU(keras.layers.Layer):
	def __init__(self, alpha=0.5, **kwargs):
		super(EPReLU, self).__init__(**kwargs)
		self.alpha = alpha

	def build(self, input_shape):
		# Trainable (PReLU) parameter
		self.a = self.add_weight(name='a', shape=input_shape[1:], dtype=K.floatx(), initializer=keras.initializers.RandomUniform(0.0, 1.0))

		# Finish building
		super(EPReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# Generate random uniform tensor between [1-alpha, 1+alpha] for training and ones tensor for test
		k = K.in_train_phase(K.random_uniform(inputs.shape[1:], 1 - self.alpha, 1 + self.alpha),
							 K.ones(inputs.shape[1:]))

		pos = keras.activations.relu(inputs) * k
		neg = -self.a * keras.activations.relu(-inputs)

		return pos + neg


class SQRTActivation(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(SQRTActivation, self).__init__(**kwargs)

	def build(self, input_shape):
		super(SQRTActivation, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = K.sqrt(keras.activations.relu(inputs))
		neg = - K.sqrt(keras.activations.relu(-inputs))

		return pos + neg


# Randomized Leaky Rectified Linear Unit
class RReLu(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(RReLu, self).__init__(**kwargs)

	def build(self, input_shape):
		# self.alpha = self.add_weight(name='alpha', shape=input_shape[1:], dtype=K.floatx(),
		#							 initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0))

		super(RReLu, self).build(input_shape)

	def call(self, inputs, **kwargs):
		# Generate random uniform alpha
		alpha = K.in_train_phase(K.random_uniform(inputs.shape[1:], 0.0, 1.0), K.constant((0.0+1.0)/2.0, shape=inputs.shape[1:]))

		pos = keras.activations.relu(inputs)
		neg = alpha * keras.activations.relu(-inputs)

		return pos + neg


class PELU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(PELU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1))
		# self.alpha = K.clip(self.alpha, 0.0001, 10)

		self.beta = self.add_weight(name='beta', shape=(1,), dtype=K.floatx(),
									initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1))
		# self.beta = K.clip(self.beta, 0.0001, 10)

		super(PELU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = (cons_greater_zero(self.alpha) / cons_greater_zero(self.beta)) * keras.activations.relu(inputs)
		neg = cons_greater_zero(self.alpha) * (K.exp((-keras.activations.relu(-inputs)) / cons_greater_zero(self.beta)) - 1)

		return pos + neg


class SlopedReLU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(SlopedReLU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=1.0, maxval=10.0))
		self.alpha = K.clip(self.alpha, 1.0, 10)

		super(SlopedReLU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return keras.activations.relu(self.alpha * inputs)


class PTELU(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(PTELU, self).__init__(**kwargs)

	def build(self, input_shape):
		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1))
		self.alpha = K.clip(self.alpha, 0.0001, 100)

		self.beta = self.add_weight(name='beta', shape=(1,), dtype=K.floatx(),
									initializer=keras.initializers.RandomUniform(minval=0.01, maxval=1))
		self.beta = K.clip(self.beta, 0.0001, 100)

		super(PTELU, self).build(input_shape)

	def call(self, inputs, **kwargs):
		pos = keras.activations.relu(inputs)
		neg = self.alpha * K.tanh(- self.beta * keras.activations.relu(-inputs))

		return pos + neg

class ELUPlus(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(ELUPlus, self).__init__(**kwargs)

	def build(self, input_shape):
		self.lmbd = self.add_weight(name='lambda', shape=(int(input_shape[-1]),), dtype=K.floatx(),
									 initializer=keras.initializers.Constant(value=0.5))

		self.lmbd = K.clip(self.lmbd, 0.0, 1.0)

		super(ELUPlus, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return self.lmbd * K.elu(inputs) + (1 - self.lmbd) * K.softplus(inputs)


class ELUSPPT(keras.layers.Layer):
	def __init__(self, **kwargs):
		super(ELUSPPT, self).__init__(**kwargs)

	def build(self, input_shape):
		self.lmbd = self.add_weight(name='lambda', shape=(int(input_shape[-1]),), dtype=K.floatx(),
									 initializer=keras.initializers.Constant(value=0.5))

		self.lmbd = K.clip(self.lmbd, 0.0, 1.0)

		self.alpha = self.add_weight(name='alpha', shape=(1,), dtype=K.floatx(),
									 initializer=keras.initializers.RandomUniform(minval=0, maxval=1),
									 trainable=True)

		super(ELUSPPT, self).build(input_shape)

	def call(self, inputs, **kwargs):
		output = self.lmbd * K.elu(inputs) + (1 - self.lmbd) * (K.softplus(inputs) - self.alpha)
		return output


# CIFAR-10 c = 1, k = 2
# CIFAR-100 c = 2, k = 1
# MNIST c = 100, k = 1.2
# CINIC-10 c = 1, k = 2
# Fashion-MNIST c = 100, k = 1.2
class Softplusplus(keras.layers.Layer):
	def __init__(self, c = 1.0, k = 2.0, **kwargs):
		super(Softplusplus, self).__init__(**kwargs)
		self.c = c
		self.k = k

	def build(self, input_shape):
		super(Softplusplus, self).build(input_shape)

	def call(self, inputs, **kwargs):
		return tf.math.softplus(self.k * inputs) + (inputs / self.c) - tf.math.log(2.0)
		# return tf.math.log(1.0 + tf.math.exp(self.k * inputs)) + (inputs / self.c) - tf.math.log(2.0)