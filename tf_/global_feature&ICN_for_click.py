
from collections import namedtuple, OrderedDict
from copy import copy
import itertools
from itertools import chain
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras import optimizers


import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Input, Layer, Embedding, Lambda
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.backend import batch_dot
from tensorflow.python.keras.initializers import RandomNormal, Zeros, glorot_normal, glorot_uniform, TruncatedNormal
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.layers import utils

######################################################################################
######################################################################################
#########################               activation         ###########################
######################################################################################
######################################################################################
######################################################################################
try:
    unicode
except NameError:
    unicode = str


class Dice(Layer):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.
      Input shape
        - Arbitrary. Use the keyword argument `input_shape` (tuple of integers, does not include the samples axis) when using this layer as the first layer in a model.
      Output shape
        - Same shape as the input.
      Arguments
        - **axis** : Integer, the axis that should be used to compute data distribution (typically the features axis).
        - **epsilon** : Small float added to variance to avoid dividing by zero.
      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(self, axis=-1, epsilon=1e-9, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(shape=(input_shape[-1],), initializer=Zeros(
        ), dtype=tf.float32, name='dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs, training=None, **kwargs):
        inputs_normed = self.bn(inputs, training=training)
        # tf.layers.batch_normalization(
        # inputs, axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self, ):
        config = {'axis': self.axis, 'epsilon': self.epsilon}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def activation_layer(activation):
    if activation in ("dice", "Dice"):
        act_layer = Dice()
    elif isinstance(activation, (str, unicode)):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer

######################################################################################
######################################################################################
#########################              layers.core         ###########################
######################################################################################
######################################################################################
######################################################################################
class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, output_activation=None,
                 seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)

        output = tf.reshape(x, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'task': self.task, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
######################################################################################
######################################################################################
#########################          layers.interaction      ###########################
######################################################################################
######################################################################################
######################################################################################
class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
    """

    def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.l2_reg = l2_reg
        self.seed = seed
        print('CrossNet parameterization:', self.parameterization)
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])
        if self.parameterization == 'vector':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, 1),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        elif self.parameterization == 'matrix':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, dim),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
                dot_ = tf.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)  # W * xi  (bs, dim, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'layer_num': self.layer_num, 'parameterization': self.parameterization,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNet, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class CrossNetMix(Layer):
    """The Cross Network part of DCN-Mix model, which improves DCN-M by:
      1 add MOE to learn feature interactions in different subspaces
      2 add nonlinear transformations in low-dimensional space
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **low_rank** : Positive integer, dimensionality of low-rank sapce.
        - **num_experts** : Positive integer, number of experts.
        - **layer_num**: Positive integer, the cross layer number
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, low_rank=32, num_experts=4, layer_num=2, l2_reg=0, seed=1024, **kwargs):
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.layer_num = layer_num
        self.l2_reg = l2_reg
        self.seed = seed
        super(CrossNetMix, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])

        # U: (dim, low_rank)
        self.U_list = [self.add_weight(name='U_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]
        # V: (dim, low_rank)
        self.V_list = [self.add_weight(name='V_list' + str(i),
                                       shape=(self.num_experts, dim, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]
        # C: (low_rank, low_rank)
        self.C_list = [self.add_weight(name='C_list' + str(i),
                                       shape=(self.num_experts, self.low_rank, self.low_rank),
                                       initializer=glorot_normal(
                                           seed=self.seed),
                                       regularizer=l2(self.l2_reg),
                                       trainable=True) for i in range(self.layer_num)]

        self.gating = [tf.keras.layers.Dense(1, use_bias=False) for i in range(self.num_experts)]

        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNetMix, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](tf.squeeze(x_l, axis=2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = tf.einsum('ij,bjk->bik', tf.transpose(self.V_list[i][expert_id]), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = tf.nn.tanh(v_x)
                v_x = tf.einsum('ij,bjk->bik', self.C_list[i][expert_id], v_x)  # (bs, low_rank, 1)
                v_x = tf.nn.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = tf.einsum('ij,bjk->bik', self.U_list[i][expert_id], v_x)  # (bs, dim, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(tf.squeeze(dot_, axis=2))

            # (3) mixture of low-rank experts
            output_of_experts = tf.stack(output_of_experts, 2)  # (bs, dim, num_experts)
            gating_score_of_experts = tf.stack(gating_score_of_experts, 1)  # (bs, num_experts, 1)
            moe_out = tf.matmul(output_of_experts, tf.nn.softmax(gating_score_of_experts, 1))
            x_l = moe_out + x_l  # (bs, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

    def get_config(self, ):

        config = {'low_rank': self.low_rank, 'num_experts': self.num_experts, 'layer_num': self.layer_num,
                  'l2_reg': self.l2_reg, 'seed': self.seed}
        base_config = super(CrossNetMix, self).get_config()
        base_config.update(config)
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape


class FM(Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d,\
                             expect to be 3 dimensions" % (len(input_shape)))

        super(FM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (K.ndim(inputs)))

        concated_embeds_value = inputs

        square_of_sum = tf.square(reduce_sum(
            concated_embeds_value, axis=1, keep_dims=True))
        sum_of_square = reduce_sum(
            concated_embeds_value * concated_embeds_value, axis=1, keep_dims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * reduce_sum(cross_term, axis=2, keep_dims=False)

        return cross_term

    def compute_output_shape(self, input_shape):
        return (None, 1)


######################################################################################
######################################################################################
#########################          layers.sequence         ###########################
######################################################################################
######################################################################################
######################################################################################
class SequencePoolingLayer(Layer):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.
      Input shape
        - A list of two  tensor [seq_value,seq_len]
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.
      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.
        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, mode='mean', supports_masking=False, **kwargs):

        if mode not in ['sum', 'mean', 'max']:
            raise ValueError("mode must be sum or mean")
        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        super(SequencePoolingLayer, self).__init__(**kwargs)

        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, seq_value_len_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            uiseq_embed_list = seq_value_len_list
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        if self.mode == "max":
            hist = uiseq_embed_list - (1 - mask) * 1e9
            return reduce_max(hist, 1, keep_dims=True)

        hist = reduce_sum(uiseq_embed_list * mask, 1, keep_dims=False)

        if self.mode == "mean":
            hist = div(hist, tf.cast(user_behavior_length, tf.float32) + self.eps)

        hist = tf.expand_dims(hist, axis=1)
        return hist

    def compute_output_shape(self, input_shape):
        if self.supports_masking:
            return (None, 1, input_shape[-1])
        else:
            return (None, 1, input_shape[0][-1])

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'supports_masking': self.supports_masking}
        base_config = super(SequencePoolingLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedSequenceLayer(Layer):
    """The WeightedSequenceLayer is used to apply weight score on variable-length sequence feature/multi-value feature.
      Input shape
        - A list of two  tensor [seq_value,seq_len,seq_weight]
        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``
        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.
        - seq_weight is a 3D tensor with shape: ``(batch_size, T, 1)``
      Output shape
        - 3D tensor with shape: ``(batch_size, T, embedding_size)``.
      Arguments
        - **weight_normalization**: bool.Whether normalize the weight score before applying to sequence.
        - **supports_masking**:If True,the input need to support masking.
    """

    def __init__(self, weight_normalization=True, supports_masking=False, **kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)
        self.weight_normalization = weight_normalization
        self.supports_masking = supports_masking

    def build(self, input_shape):
        if not self.supports_masking:
            self.seq_len_max = int(input_shape[0][1])
        super(WeightedSequenceLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, input_list, mask=None, **kwargs):
        if self.supports_masking:
            if mask is None:
                raise ValueError(
                    "When supports_masking=True,input must support masking")
            key_input, value_input = input_list
            mask = tf.expand_dims(mask[0], axis=2)
        else:
            key_input, key_length_input, value_input = input_list
            mask = tf.sequence_mask(key_length_input,
                                    self.seq_len_max, dtype=tf.bool)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = key_input.shape[-1]

        if self.weight_normalization:
            paddings = tf.ones_like(value_input) * (-2 ** 32 + 1)
        else:
            paddings = tf.zeros_like(value_input)
        value_input = tf.where(mask, value_input, paddings)

        if self.weight_normalization:
            value_input = softmax(value_input, dim=1)

        if len(value_input.shape) == 2:
            value_input = tf.expand_dims(value_input, axis=2)
            value_input = tf.tile(value_input, [1, 1, embedding_size])

        return tf.multiply(key_input, value_input)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask):
        if self.supports_masking:
            return mask[0]
        else:
            return None

    def get_config(self, ):
        config = {'weight_normalization': self.weight_normalization, 'supports_masking': self.supports_masking}
        base_config = super(WeightedSequenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

######################################################################################
######################################################################################
##########################              input             ############################
######################################################################################
######################################################################################
######################################################################################
def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(fg.vocabulary_size, mask_zero=(feat_name in mask_feat_list))(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):
    from . import feature_column as fc_lib

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.VarLenSparseFeat), feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = fc.length_name
        if feature_length_name is not None:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm)(
                    [embedding_dict[feature_name], features[feature_length_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [seq_input, features[feature_length_name]])
        else:
            if fc.weight_name is not None:
                seq_input = WeightedSequenceLayer(weight_normalization=fc.weight_norm, supports_masking=True)(
                    [embedding_dict[feature_name], features[fc.weight_name]])
            else:
                seq_input = embedding_dict[feature_name]
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
                seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(features, feature_columns):
    from . import feature_column as fc_lib
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, fc_lib.DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        if fc.transform_fn is None:
            dense_input_list.append(features[fc.name])
        else:
            transform_result = Lambda(fc.transform_fn)(features[fc.name])
            dense_input_list.append(transform_result)
    return dense_input_list


def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c

######################################################################################
######################################################################################
##########################           layer.util           ############################
######################################################################################
######################################################################################
######################################################################################

class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None


class Hash(tf.keras.layers.Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Hash, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):


        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x, )
        else:
            zero = tf.as_string(tf.zeros([1], dtype='int32'))

        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, num_buckets,
                                                   name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x
    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero, }
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, use_bias=False, seed=1024, **kwargs):

        self.l2_reg = l2_reg
        # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
        if mode not in [0, 1, 2]:
            raise ValueError("mode must be 0,1 or 2")
        self.mode = mode
        self.use_bias = use_bias
        self.seed = seed
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        if self.mode == 1:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(self.seed),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)
        elif self.mode == 2:
            self.kernel = self.add_weight(
                'linear_kernel',
                shape=[int(input_shape[1][-1]), 1],
                initializer=tf.keras.initializers.glorot_normal(self.seed),
                regularizer=tf.keras.regularizers.l2(self.l2_reg),
                trainable=True)

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = fc
        else:
            sparse_input, dense_input = inputs
            fc = tf.tensordot(dense_input, self.kernel, axes=(-1, 0))
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + fc
        if self.use_bias:
            linear_logit += self.bias

        return linear_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg, 'use_bias': self.use_bias, 'seed': self.seed}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def reduce_mean(input_tensor,
                axis=None,
                keep_dims=False,
                name=None,
                reduction_indices=None):
    try:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keep_dims=keep_dims,
                              name=name,
                              reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_mean(input_tensor,
                              axis=axis,
                              keepdims=keep_dims,
                              name=name)


def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_max(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)


def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)


class Add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1:
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])

        return tf.keras.layers.add(inputs)


def add_func(inputs):
    return Add()(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


######################################################################################
######################################################################################
##########################          feature_column        ############################
######################################################################################
######################################################################################
######################################################################################

DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype', 'transform_fn'])):
    """ Dense feature
    Args:
        name: feature name,
        dimension: dimension of the feature, default = 1.
        dtype: dtype of the feature, default="float32".
        transform_fn: If not `None` , a function that can be used to transform
        values of the feature.  the function takes the input Tensor as its
        argument, and returns the output Tensor.
        (e.g. lambda x: (x - 3.0) / 4.2).
    """
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32", transform_fn=None):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype, transform_fn)

    def __hash__(self):
        return self.name.__hash__()

    # def __eq__(self, other):
    #     if self.name == other.name:
    #         return True
    #     return False

    # def __repr__(self):
    #     return 'DenseFeat:'+self.name


def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    return list(features.keys())


def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if fc.weight_name is not None:
                input_features[fc.weight_name] = Input(shape=(fc.maxlen, 1), name=prefix + fc.weight_name,
                                                       dtype="float32")
            if fc.length_name is not None:
                input_features[fc.length_name] = Input((1,), name=prefix + fc.length_name, dtype='int32')

        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features


def get_linear_logit(features, feature_columns, units=1, use_bias=False, seed=1024, prefix='linear',
                     l2_reg=0):
    linear_feature_columns = copy(feature_columns)
    for i in range(len(linear_feature_columns)):
        if isinstance(linear_feature_columns[i], SparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(embedding_dim=1,
                                                                           embeddings_initializer=Zeros())
        if isinstance(linear_feature_columns[i], VarLenSparseFeat):
            linear_feature_columns[i] = linear_feature_columns[i]._replace(
                sparsefeat=linear_feature_columns[i].sparsefeat._replace(embedding_dim=1,
                                                                         embeddings_initializer=Zeros()))

    linear_emb_list = [input_from_feature_columns(features, linear_feature_columns, l2_reg, seed,
                                                  prefix=prefix + str(i))[0] for i in range(units)]
    _, dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):

        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias, seed=seed)([sparse_input, dense_input])
        elif len(linear_emb_list[i]) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias, seed=seed)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias, seed=seed)(dense_input)
        else:
            # raise NotImplementedError
            return add_func([])
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)


def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                    seq_mask_zero=seq_mask_zero)
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, features,
                                                                 varlen_sparse_feature_columns)
    group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
    if not support_group:
        group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict, dense_value_list

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
if __name__ == "__main__":
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])


def GFICNN(linear_feature_columns, dnn_feature_columns, conv_kernel_width=(39, 1, 1), conv_filters=(21, 21, 1),
         dnn_hidden_units=(256,), l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0,
         seed=1024, task='binary'):
    """Instantiates the Convolutional Click Prediction Model architecture.
    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if len(conv_kernel_width) != len(conv_filters):
        raise ValueError(
            "conv_kernel_width must have same element with conv_filters")

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)
    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed,
                                    l2_reg=l2_reg_linear)
    
    sparse_embedding_list, _ = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                          seed, support_dense=False)
    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        seed, support_group=True)
    #change
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])
    
#     n = len(sparse_embedding_list)
    l = len(conv_filters)

    conv_input = combined_dnn_input(list(chain.from_iterable(
        group_embedding_dict.values())), dense_value_list)
#     conv_input = concat_func(sparse_embedding_list, axis=1)
    conv_result = tf.keras.layers.Lambda(
        lambda x: tf.expand_dims(x, axis=3))(conv_input)

    for i in range(1, l + 1):
        filters = conv_filters[i - 1]
        width = conv_kernel_width[i - 1]
#         k = max(1, int((1 - pow(i / l, l - i)) * n)) if i < l else 3

#         conv_result = tf.keras.layers.Conv2D(filters=filters, kernel_size=(width, 1), strides=(1, 1), padding='same',
#                                              activation='tanh', use_bias=True, )(pooling_result)
        conv_result = tf.keras.layers.Conv2D(filters=filters, kernel_size=(width, 1), strides=(1, 1), padding='valid',
                                                     activation='tanh', use_bias=True, )(conv_result)
#         pooling_result = KMaxPooling(
#             k=min(k, int(conv_result.shape[1])), axis=1)(conv_result)

#     flatten_result = tf.keras.layers.Flatten()(pooling_result)
    flatten_result = tf.keras.layers.Flatten()(conv_result)
    dnn_out = DNN(dnn_hidden_units, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout)(flatten_result)
    dnn_logit = tf.keras.layers.Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_out)

#     final_logit = add_func([dnn_logit, linear_logit])
    final_logit = add_func([dnn_logit, fm_logit, linear_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model



#   4.Define Model,train,predict and evaluate
model = GFICNN(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile(optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0001), "binary_crossentropy",
              metrics=['binary_crossentropy'], )
es = EarlyStopping(monitor='val_binary_crossentropy')
history = model.fit(train_model_input, train[target].values,
                    batch_size=500, epochs=50, verbose=2, validation_split=0.2,callbacks=[es] )
pred_ans = model.predict(test_model_input, batch_size=500)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
