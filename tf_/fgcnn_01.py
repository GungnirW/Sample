import tensorflow as tf
import sys

from collections import OrderedDict, namedtuple
from itertools import chain
import itertools

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform)
from tensorflow.python.keras.layers import Layer
from tensorflow.python.layers import utils
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import  Embedding, Input, Flatten
from tensorflow.python.keras.regularizers import l2

from collections import OrderedDict, namedtuple

from tensorflow.python.keras.initializers import RandomNormal

# 输入:feature_columns
# 输出：
# 
# 
#
# 
class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype','embedding_name','embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None,embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name,embedding)

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)


class VarLenSparseFeat(namedtuple('VarLenFeat', ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype','embedding_name','embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype, embedding_name,embedding)


'''************************************************ input *************************************************'''

'''************************************************ build_input_features *************************************************'''

def build_input_features(feature_columns, include_varlen=True, mask_zero=True, prefix='',include_fixlen=True):
    input_features = OrderedDict()
    if include_fixlen:
        for fc in feature_columns:
            if isinstance(fc,SparseFeat):
                input_features[fc.name] = Input(
                    shape=(1,), name=prefix+fc.name, dtype=fc.dtype)
            elif isinstance(fc,DenseFeat):
                input_features[fc.name] = Input(
                    shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
    if include_varlen:
        for fc in feature_columns:
            if isinstance(fc,VarLenSparseFeat):
                input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + 'seq_' + fc.name,
                                                      dtype=fc.dtype)
        if not mask_zero:
            for fc in feature_columns:
                input_features[fc.name+"_seq_length"] = Input(shape=(
                    1,), name=prefix + 'seq_length_' + fc.name)
                input_features[fc.name+"_seq_max_length"] = fc.maxlen


    return input_features

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
        self.eps = 1e-8
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
            mask = tf.cast(mask,tf.float32)#                tf.to_float(mask)
            user_behavior_length = reduce_sum(mask, axis=-1, keep_dims=True)
            mask = tf.expand_dims(mask, axis=2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list

            mask = tf.sequence_mask(user_behavior_length,
                                    self.seq_len_max, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = uiseq_embed_list.shape[-1]

        mask = tf.tile(mask, [1, 1, embedding_size])

        uiseq_embed_list *= mask
        hist = uiseq_embed_list
        if self.mode == "max":
            return reduce_max(hist, 1, keep_dims=True)

        hist = reduce_sum(hist, 1, keep_dims=False)

        if self.mode == "mean":
            hist = div(hist, user_behavior_length + self.eps)

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

'''************************************************ input *************************************************
        create_embedding_dict, create_embedding_matrix, embedding_lookup, get_varlen_pooling_list, varlen_embedding_lookup
************************************************ input_from_feature_columns *************************************************'''
def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    if embedding_size == 'auto':
        print("Notice:Do not use auto embedding in models other than DCN")
        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(l2_reg),
                                                 name=prefix + '_emb_' + feat.name) for feat in
                            sparse_feature_columns}
    else:

        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, embedding_size,
                                                 embeddings_initializer=RandomNormal(
                                                     mean=0.0, stddev=init_std, seed=seed),
                                                 embeddings_regularizer=l2(
                                                     l2_reg),
                                                 name=prefix + '_emb_'  + feat.name) for feat in
                            sparse_feature_columns}

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            if embedding_size == "auto":
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_seq_emb_' + feat.name,
                                                        mask_zero=seq_mask_zero)

            else:
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                        embeddings_initializer=RandomNormal(
                                                            mean=0.0, stddev=init_std, seed=seed),
                                                        embeddings_regularizer=l2(
                                                            l2_reg),
                                                        name=prefix + '_seq_emb_' + feat.name,
                                                        mask_zero=seq_mask_zero)


    return sparse_embedding

def create_embedding_matrix(feature_columns,l2_reg,init_std,seed,embedding_size, prefix="",seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed,
                                                 l2_reg, prefix=prefix + 'sparse',seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict

def embedding_lookup(sparse_embedding_dict,sparse_input_dict,sparse_feature_columns,return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0  or feature_name in return_feat_list and fc.embedding:
            if fc.use_hash:
                lookup_idx = Hash(fc.dimension,mask_zero=(feature_name in mask_feat_list))(sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]

            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))

    return embedding_vec_list

def get_dense_input(features,feature_columns):
    dense_feature_columns = list(filter(lambda x:isinstance(x,DenseFeat),feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list
def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.dimension, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    return varlen_embedding_vec_dict

def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns):
    pooling_vec_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = feature_name + '_seq_length'
        if feature_length_name in features:
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
            [embedding_dict[feature_name], features[feature_length_name]])
        else:
            vec = SequencePoolingLayer(combiner, supports_masking=True)(
            embedding_dict[feature_name])
        pooling_vec_list.append(vec)
    return pooling_vec_list

'''************************************************ input_from_feature_columns *************************************************'''
def input_from_feature_columns(features,feature_columns, embedding_size, l2_reg, init_std, seed,prefix='',seq_mask_zero=True,support_dense=True):


    sparse_feature_columns = list(filter(lambda x:isinstance(x,SparseFeat),feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    embedding_dict = create_embedding_matrix(feature_columns,l2_reg,init_std,seed,embedding_size, prefix=prefix,seq_mask_zero=seq_mask_zero)
    sparse_embedding_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features,feature_columns)
    if not support_dense and len(dense_value_list) >0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict,features,varlen_sparse_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, varlen_sparse_feature_columns)
    sparse_embedding_list += sequence_embed_list

    return sparse_embedding_list, dense_value_list


'''************************************************ PredictionLayer *************************************************'''

'''************************************************ PredictionLayer *************************************************'''
#目前来看，这层就是sigmoid函数 = 1/（1+e^-x）

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
		
'''************************************************ DNN *************************************************'''

'''************************************************ DNN *************************************************'''
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
        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
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

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate,seed=self.seed+i) for i in range(len(self.hidden_units))]

        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            # fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc,training = training)
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
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""**********************InnerProductLayer**********************"""


"""**********************InnerProductLayer**********************"""
class InnerProductLayer(Layer):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.
      Input shape
        - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape: ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.
      Arguments
        - **reduce_sum**: bool. Whether return inner product or element-wise product
      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, reduce_sum=True, **kwargs):
        self.reduce_sum = reduce_sum
        super(InnerProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `InnerProductLayer` layer should be called '
                             'on a list of at least 2 inputs')

        reduced_inputs_shapes = [shape.as_list() for shape in input_shape]
        shape_set = set()

        for i in range(len(input_shape)):
            shape_set.add(tuple(reduced_inputs_shapes[i]))

        if len(shape_set) > 1:
            raise ValueError('A `InnerProductLayer` layer requires '
                             'inputs with same shapes '
                             'Got different shapes: %s' % (shape_set))

        if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
            raise ValueError('A `InnerProductLayer` layer requires '
                             'inputs of a list with same shape tensor like (None,1,embedding_size)'
                             'Got different shapes: %s' % (input_shape[0]))
        super(InnerProductLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([embed_list[idx]
                       for idx in row], axis=1)  # batch num_pairs k
        q = tf.concat([embed_list[idx]
                       for idx in col], axis=1)

        inner_product = p * q
        if self.reduce_sum:
            inner_product = reduce_sum(
                inner_product, axis=2, keep_dims=True)
        return inner_product

    def compute_output_shape(self, input_shape):
        num_inputs = len(input_shape)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        input_shape = input_shape[0]
        embed_size = input_shape[-1]
        if self.reduce_sum:
            return (input_shape[0], num_pairs, 1)
        else:
            return (input_shape[0], num_pairs, embed_size)

    def get_config(self, ):
        config = {'reduce_sum': self.reduce_sum, }
        base_config = super(InnerProductLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""**********************Feature Generation Layer used in FGCNN,including Convolution,MaxPooling and Recombination.**********************"""


"""**********************Feature Generation Layer used in FGCNN,including Convolution,MaxPooling and Recombination.**********************"""
class FGCNNLayer(Layer):
    """Feature Generation Layer used in FGCNN,including Convolution,MaxPooling and Recombination.
      Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,new_feture_num,embedding_size)``.
      References
        - [Liu B, Tang R, Chen Y, et al. Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1904.04447, 2019.](https://arxiv.org/pdf/1904.04447)
    """

    def __init__(self, filters=(14, 16,), kernel_width=(7, 7,), new_maps=(3, 3,), pooling_width=(2, 2),
                 **kwargs):
        if not (len(filters) == len(kernel_width) == len(new_maps) == len(pooling_width)):
            raise ValueError("length of argument must be equal")
        self.filters = filters
        self.kernel_width = kernel_width
        self.new_maps = new_maps
        self.pooling_width = pooling_width

        super(FGCNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))
        self.conv_layers = []
        self.pooling_layers = []
        self.dense_layers = []
        pooling_shape = input_shape.as_list() + [1, ]
        embedding_size = int(input_shape[-1])
        for i in range(1, len(self.filters) + 1):       #range范围不包括end
            filters = self.filters[i - 1]
            width = self.kernel_width[i - 1]
            new_filters = self.new_maps[i - 1]          #?new??????????????????????????????????
            pooling_width = self.pooling_width[i - 1]
            conv_output_shape = self._conv_output_shape(
                pooling_shape, (width, 1))
            pooling_shape = self._pooling_output_shape(
                conv_output_shape, (pooling_width, 1))
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=filters, kernel_size=(width, 1), strides=(1, 1),
                                                           padding='same',
                                                           activation='tanh', use_bias=True, ))
            self.pooling_layers.append(
                tf.keras.layers.MaxPooling2D(pool_size=(pooling_width, 1)))
                                                           
                                                         #输出的维度,dense(dim_output, , )
            self.dense_layers.append(tf.keras.layers.Dense(pooling_shape[1] * embedding_size * new_filters,
                                                           activation='tanh', use_bias=True))

        self.flatten = tf.keras.layers.Flatten()

        super(FGCNNLayer, self).build(
            input_shape)  # Be sure to call this somewhere!

    #call和init相关联，在call里实现模型的正向传递
    def call(self, inputs, **kwargs):

        #backend.ndim()返回轴数
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        embedding_size = int(inputs.shape[-1])
        pooling_result = tf.expand_dims(inputs, axis=3)

        new_feature_list = []

        for i in range(1, len(self.filters) + 1):
            new_filters = self.new_maps[i - 1]

            conv_result = self.conv_layers[i - 1](pooling_result)

            pooling_result = self.pooling_layers[i - 1](conv_result)

            flatten_result = self.flatten(pooling_result)

            #重组层，
            new_result = self.dense_layers[i - 1](flatten_result)

            new_feature_list.append(
                tf.reshape(new_result, (-1, int(pooling_result.shape[1]) * new_filters, embedding_size)))
        
        #new_features are ??????????????????????reconstract
        new_features = concat_fun(new_feature_list, axis=1)
        return new_features

    # 当改变了输入张量的shape，得在这里定义修改如何去计算各层的shape
    def compute_output_shape(self, input_shape):

        new_features_num = 0
        features_num = input_shape[1]

        for i in range(0, len(self.kernel_width)):
            pooled_features_num = features_num // self.pooling_width[i]
            new_features_num += self.new_maps[i] * pooled_features_num
            features_num = pooled_features_num

        return (None, new_features_num, input_shape[-1])

    def get_config(self, ):
        config = {'kernel_width': self.kernel_width, 'filters': self.filters, 'new_maps': self.new_maps,
                  'pooling_width': self.pooling_width}
        base_config = super(FGCNNLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _conv_output_shape(self, input_shape, kernel_size):
        # channels_last
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = utils.conv_output_length(
                space[i],
                kernel_size[i],
                padding='same',
                stride=1,
                dilation=1)
            new_space.append(new_dim)
        return ([input_shape[0]] + new_space + [self.filters])

    def _pooling_output_shape(self, input_shape, pool_size):
        # channels_last

        rows = input_shape[1]
        cols = input_shape[2]
        rows = utils.conv_output_length(rows, pool_size[0], 'valid',
                                        pool_size[0])
        cols = utils.conv_output_length(cols, pool_size[1], 'valid',
                                        pool_size[1])
        return [input_shape[0], rows, cols, input_shape[3]]



'''************************************************ Utils *************************************************'''
'''************************************************ Utils *************************************************
                    
                    cat_fun,reduce_sum,reduce_max,div,softmax,reduce_mean,Hash,Linear


************************************************ Utils *************************************************'''
'''************************************************ Utils *************************************************'''
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
            x = tf.as_string(x, )
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                    name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                               name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x

    def compute_mask(self, inputs, mask):
        return None

    def get_config(self, ):
        config = {'num_buckets': self.num_buckets, 'mask_zero': self.mask_zero}
        base_config = super(Hash, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Linear(tf.keras.layers.Layer):

    def __init__(self, l2_reg=0.0, mode=0, **kwargs):

        self.l2_reg = l2_reg
        # self.l2_reg = tf.contrib.layers.l2_regularizer(float(l2_reg_linear))
        self.mode = mode
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bias = self.add_weight(name='linear_bias',
                                    shape=(1,),
                                    initializer=tf.keras.initializers.Zeros(),
                                    trainable=True)

        self.dense = tf.keras.layers.Dense(units=1, activation=None, use_bias=False,
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))

        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs , **kwargs):

        if self.mode == 0:
            sparse_input = inputs
            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=True)
        elif self.mode == 1:
            dense_input = inputs
            linear_logit = self.dense(dense_input)

        else:
            sparse_input, dense_input = inputs

            linear_logit = reduce_sum(sparse_input, axis=-1, keep_dims=False) + self.dense(dense_input)

        linear_bias_logit = linear_logit + self.bias

        return linear_bias_logit

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'mode': self.mode, 'l2_reg': self.l2_reg}
        base_config = super(Linear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)
def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    if tf.__version__ < '2.0.0':
        return tf.reduce_sum(input_tensor,
                   axis=axis,
                   keep_dims=keep_dims,
                   name=name,
                   reduction_indices=reduction_indices)
    else:
        return  tf.reduce_sum(input_tensor,
                   axis=axis,
                   keepdims=keep_dims,
                   name=name)

def reduce_max(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    if tf.__version__ < '2.0.0':
        return tf.reduce_max(input_tensor,
                   axis=axis,
                   keep_dims=keep_dims,
                   name=name,
                   reduction_indices=reduction_indices)
    else:
        return  tf.reduce_max(input_tensor,
                   axis=axis,
                   keepdims=keep_dims,
                   name=name)

def div(x, y, name=None):
    if tf.__version__ < '2.0.0':
        return tf.div(x, y, name=name)
    else:
        return tf.divide(x, y, name=name)

def softmax(logits, dim=-1, name=None):
    if tf.__version__ < '2.0.0':
        return tf.nn.softmax(logits, dim=dim, name=name)
    else:
        return tf.nn.softmax(logits, axis=dim, name=name)


'''************************************************ activation_layer *************************************************'''
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
        ), dtype=tf.float32, name= 'dice_alpha')  # name='alpha_'+self.name
        super(Dice, self).build(input_shape)  # Be sure to call this somewhere!
        self.uses_learning_phase = True

    def call(self, inputs,training=None,**kwargs):
        inputs_normed = self.bn(inputs,training=training)
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
    if activation == "dice" or activation == "Dice":
        act_layer =  Dice()
    elif (isinstance(activation, str)) or (sys.version_info.major == 2 and isinstance(activation, (str, unicode))):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return act_layer



'''************************************************ FGCNN *************************************************'''
'''************************************************ FGCNN *************************************************'''
def unstack(input_tensor):
    input_ = tf.expand_dims(input_tensor, axis=2)#增加第三维度
    return tf.unstack(input_, input_.shape[1], 1)#矩阵第二维度分解


def FGCNN(dnn_feature_columns, embedding_size=8, conv_kernel_width=(7, 7, 7, 7), conv_filters=(14, 16, 18, 20),
          new_maps=(3, 3, 3, 3),
          pooling_width=(2, 2, 2, 2), dnn_hidden_units=(128,), l2_reg_embedding=1e-5, l2_reg_dnn=0, dnn_dropout=0,
          init_std=0.0001, seed=1024,
          task='binary', ):
    """Instantiates the Feature Generation by Convolutional Neural Network architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.    
    :param embedding_size: positive integer,sparse feature embedding_size
    :param conv_kernel_width: list,list of positive integer or empty list,the width of filter in each conv layer.
    :param conv_filters: list,list of positive integer or empty list,the number of filters in each conv layer.
    :param new_maps: list, list of positive integer or empty list, the feature maps of generated features.
    :param pooling_width: list, list of positive integer or empty list,the width of pooling layer.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net.
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    if not (len(conv_kernel_width) == len(conv_filters) == len(new_maps) == len(pooling_width)):
        raise ValueError(
            "conv_kernel_width,conv_filters,new_maps  and pooling_width must have same length")

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    #
    #fg_deep_emb_list作为fgcnn的输入，features:
    # 
    # 
    #
    deep_emb_list, _ = input_from_feature_columns(features,dnn_feature_columns,
                                                                              embedding_size,
                                                                              l2_reg_embedding,init_std,
                                                                              seed)
    fg_deep_emb_list,_ = input_from_feature_columns(features,dnn_feature_columns,
                                                                              embedding_size,
                                                                              l2_reg_embedding,init_std,
                                                                              seed,prefix='fg')
 
    #3d
    fg_input = concat_fun(fg_deep_emb_list, axis=1)
    origin_input = concat_fun(deep_emb_list, axis=1)

    if len(conv_filters) > 0:
        #fgcnn融合特征
        new_features = FGCNNLayer(
            conv_filters, conv_kernel_width, new_maps, pooling_width)(fg_input)
        combined_input = concat_fun([origin_input, new_features], axis=1)
    #combined_input为cnnfeature和rawdata的结合
    else:                           
        combined_input = origin_input
    
    #IPNN层
    inner_product = tf.keras.layers.Flatten()(InnerProductLayer()(
        tf.keras.layers.Lambda(unstack, mask=[None] * int(combined_input.shape[1]))(combined_input)))
    #转为一维linear——signal
    linear_signal = tf.keras.layers.Flatten()(combined_input)
    dnn_input = tf.keras.layers.Concatenate()([linear_signal, inner_product])
    dnn_input = tf.keras.layers.Flatten()(dnn_input)

    final_logit = DNN(dnn_hidden_units, dropout_rate=dnn_dropout,
                      l2_reg=l2_reg_dnn)(dnn_input)
    
    #dense全连接：unit：dimension of outputs
    final_logit = tf.keras.layers.Dense(1, use_bias=False)(final_logit)

    #task = binary logloss
    #final_logit
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model