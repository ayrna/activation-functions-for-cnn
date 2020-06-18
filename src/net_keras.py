import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense, Lambda
from keras import regularizers
from keras import backend as K
from keras.initializers import glorot_uniform, random_normal
from activations import SPP, SPPT, MPELU, RTReLU, RTPReLU, PairedReLU, EReLU, SQRTActivation, RReLu, PELU, SlopedReLU, PTELU, EPReLU, ELUPlus, ELUSPPT, Softplusplus


class Net:
	def __init__(self, size, activation, final_activation, f_a_params={}, use_tau=True, prob_layer=None, num_channels=3,
				 num_classes=5, spp_alpha=0.2, dropout=0):
		self.size = size
		self.activation = activation
		self.final_activation = final_activation
		self.f_a_params = f_a_params
		self.use_tau = use_tau
		self.prob_layer = prob_layer
		self.num_channels = num_channels
		self.num_classes = num_classes
		self.spp_alpha = spp_alpha
		self.dropout = dropout

	def build(self, net_model):
		if hasattr(self, net_model):
			return getattr(self, net_model)()
		else:
			raise Exception('Invalid network model.')

	def vgg16(self):
		# Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
		model = Sequential()
		weight_decay = 0.0005

		model.add(Conv2D(64, (3, 3), padding='same',
						 input_shape=(self.size, self.size, self.num_channels), kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.3, seed=1))

		model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4, seed=1))

		model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4, seed=1))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4, seed=1))

		model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4, seed=1))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4, seed=1))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4, seed=1))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())
		model.add(Dropout(0.4, seed=1))

		model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=1), kernel_regularizer=regularizers.l2(weight_decay)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5, seed=1))

		model.add(Flatten())
		model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=glorot_uniform(seed=1)))
		model.add(self.__activation())
		model.add(BatchNormalization())

		model.add(Dropout(0.5, seed=1))
		model.add(Dense(self.num_classes, kernel_initializer=glorot_uniform(seed=1)))
		model.add(Activation('softmax'))

		return model

	def __activation(self):
		if self.activation == 'relu':
			return keras.layers.Activation('relu')
		elif self.activation == 'lrelu':
			return keras.layers.LeakyReLU()
		elif self.activation == 'prelu':
			return keras.layers.PReLU()
		elif self.activation == 'elu':
			return keras.layers.ELU()
		elif self.activation == 'softplus':
			return keras.layers.Activation('softplus')
		elif self.activation == 'spp':
			return SPP(self.spp_alpha)
		elif self.activation == 'sppt':
			return SPPT()
		elif self.activation == 'mpelu':
			return MPELU(channel_wise=True)
		elif self.activation == 'rtrelu':
			return RTReLU()
		elif self.activation == 'rtprelu':
			return RTPReLU()
		elif self.activation == 'pairedrelu':
			return PairedReLU()
		elif self.activation == 'erelu':
			return EReLU()
		elif self.activation == 'eprelu':
			return EPReLU()
		elif self.activation == 'sqrt':
			return SQRTActivation()
		elif self.activation == 'rrelu':
			return RReLu()
		elif self.activation == 'pelu':
			return PELU()
		elif self.activation == 'slopedrelu':
			return SlopedReLU()
		elif self.activation == 'ptelu':
			return PTELU()
		elif self.activation == 'eluplus':
			return ELUPlus()
		elif self.activation == 'elusppt':
			return ELUSPPT()
		elif self.activation == 'soft++C-10':
			return Softplusplus(c = 1.0, k = 2.0)
		elif self.activation == 'soft++C-100':
			return Softplusplus(c = 2.0, k = 1.0)
		elif self.activation == 'soft++MNIST':
			return Softplusplus(c = 100.0, k = 1.2)
		else:
			return keras.layers.Activation('relu')

	def __final_activation(self, x):
		x = keras.layers.Dense(self.num_classes)(x)
		x = keras.layers.Activation(self.final_activation)(x)
		return x