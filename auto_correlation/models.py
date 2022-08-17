import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class StrideDilationNetDetail(tf.keras.Model):
    def __init__(self,
                 list_stride=(3, 3),
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 target_size=24,
                 dropout_rate=0.0):
        self.nb_filters = nb_filters

        super(StrideDilationNetDetail, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        # D0
        self.conv1 = layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   strides=list_stride[0],
                                   padding=padding,
                                   name='conv1D_1',
                                   kernel_initializer=init)

        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

        # D1
        self.conv2 = layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   strides=list_stride[1],
                                   padding=padding,
                                   name='conv1D_2',
                                   kernel_initializer=init)

        self.batch2 = layers.BatchNormalization(axis=-1)
        self.ac2 = layers.Activation('relu')
        self.drop2 = layers.Dropout(rate=dropout_rate)

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')

        self.dense = layers.Dense(units=target_size)

        # add this code to show input shape details
        # self.call(layers.Input(shape=(history_len, 1)))

    def call(self, inputs, training=True):
        prev_x = inputs
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x) if training else x

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ac2(x)
        x = self.drop2(x) if training else x

        # if prev_x.shape[-1] != x.shape[-1]:  # match the dimention
        #     prev_x = self.downsample(prev_x)
        # assert prev_x.shape == x.shape

        # return self.ac3(prev_x + x)  # skip connection
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x

    # @property
    # def receptive_field(self):
    #     return history_len * np.prod(self.list_stride) - np.prod(self.list_stride) - \
    #            self.list_stride[0] * (1 - self.kernel_size) + self.kernel_size


class StrideLayer(layers.Layer):
    def __init__(self,
                 nb_stride=3,
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 dropout_rate=0.0,
                 init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                 name="DilatedLayer", **kwargs):
        super(StrideLayer, self).__init__(name=name, **kwargs)

        self.conv1 = layers.Conv1D(filters=nb_filters,
                                   kernel_size=kernel_size,
                                   strides=nb_stride,
                                   padding=padding,
                                   name='conv1D',
                                   kernel_initializer=init)

        self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.ac1(x)
        x = self.drop1(x)
        return x


class StrideDilatedNet(tf.keras.Model):
    def __init__(self,
                 list_stride=(3, 3),
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 target_size=24,
                 dropout_rate=0.0,
                 **kwargs):
        self.nb_filters = nb_filters
        self.list_stride = list_stride
        self.kernel_size = kernel_size
        self.padding = padding

        super(StrideDilatedNet, self).__init__(**kwargs)
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        # self.dilation1 = StrideLayer(nb_stride=list_stride[0],
        #                              nb_filters=nb_filters,
        #                              kernel_size=kernel_size,
        #                              padding=padding,
        #                              init=init,
        #                              dropout_rate=dropout_rate,
        #                              name='DilatedLayer_1')
        #
        # self.dilation2 = StrideLayer(nb_stride=list_stride[1],
        #                              nb_filters=nb_filters,
        #                              kernel_size=kernel_size,
        #                              padding=padding,
        #                              init=init,
        #                              dropout_rate=dropout_rate,
        #                              name='DilatedLayer_2')

        self.stride_blocks = []

        for i, d in enumerate(self.list_stride):
            stride_block_filters = self.nb_filters
            self.stride_blocks.append(
                StrideLayer(nb_stride=self.list_stride[i],
                            nb_filters=stride_block_filters,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            init=init,
                            dropout_rate=dropout_rate,
                            name=f"DilatedLayer_{i}")
            )

        # if prev_x.shape[-1] != x.shape[-1]:  # match the dimention
        #     prev_x = self.downsample(prev_x)
        # assert prev_x.shape == x.shape
        #
        # return self.ac3(prev_x + x)  # skip connection

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')

        self.dense = layers.Dense(units=target_size)

        # add this code to show input shape details
        # self.call(layers.Input(shape=(history_len, 1)))


    def call(self, inputs, training=True):
        x = inputs
        for stride_block in self.stride_blocks:
            x = stride_block(x)
        # x = self.dilation1(inputs)
        # x = self.dilation2(x)
        x = self.slicer_layer(x)
        x = self.dense(x)
        return x

# class DelayedLayer(layers.Layer):
#     def __init__(self,
#                  nb_stride=3,
#                  nb_filters=64,
#                  kernel_size=3,
#                  padding='causal',
#                  dropout_rate=0.0,
#                  init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
#                  name="DilatedLayer", **kwargs):
#         super(DilatedLayer, self).__init__(name=name, **kwargs)
#
#         self.conv1 = layers.Conv1D(filters=64,
#                                    kernel_size=kernel_size,
#                                    strides=nb_stride,
#                                    padding=padding,
#                                    name='conv1D',
#                                    kernel_initializer=init)
#
#         self.conv2 = layers.Conv1D(filters=64,
#                                    kernel_size=kernel_size,
#                                    strides=nb_stride,
#                                    padding=padding,
#                                    name='conv1D',
#                                    kernel_initializer=init)
#
#         self.batch1 = layers.BatchNormalization(axis=-1)
#         self.ac1 = layers.Activation('relu')
#         self.drop1 = layers.Dropout(rate=dropout_rate)
#
#     def call(self, inputs):
#         # x1: Delayed Conv -> Norm -> Dropout (x2).
#         # x2: Dilated Conv -> Norm -> D
#         # ropout (x2).
#         # x2: Residual (1x1 matching conv - optional).
#         # Output: x1 + x2.
#         # x1 -> connected to skip connections.
#         # x1 + x2 -> connected to the next block.
#         #       input
#         #     x1      x2
#         #   conv1D    1x1 Conv1D (optional)
#         #    ...
#         #   conv1D
#         #    ...
#         #       x1 + x2
#         x1 = self.conv1(inputs)
#         x2 = self.conv1(inputs)
#         x3 = concat(x1, x2)
#
#         x = self.batch1(x)
#         x = self.ac1(x)
#         x = self.drop1(x)
#         return x
#
#
# class DelayedNet(tf.keras.Model):
#     def __init__(self,
#                  list_stride=(3, 3),
#                  nb_filters=64,
#                  kernel_size=3,
#                  padding='causal',
#                  target_size=24,
#                  dropout_rate=0.0):
#         self.nb_filters = nb_filters
#         self.list_stride = list_stride
#         self.kernel_size = kernel_size
#
#         super(StrideDilatedNet, self).__init__()
#         init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
#         assert padding in ['causal', 'same']
#
#         self.dilation1 = DilatedLayer(nb_stride=list_stride[0],
#                                       nb_filters=nb_filters,
#                                       kernel_size=kernel_size,
#                                       padding=padding,
#                                       init=init,
#                                       dropout_rate=dropout_rate,
#                                       name='DilatedLayer_1')
#
#         self.dilation2 = DilatedLayer(nb_stride=list_stride[1],
#                                       nb_filters=nb_filters,
#                                       kernel_size=kernel_size,
#                                       padding=padding,
#                                       init=init,
#                                       dropout_rate=dropout_rate,
#                                       name='DilatedLayer_2')
#
#         self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')
#
#         self.dense = layers.Dense(units=target_size)
#
#         # add this code to show input shape details
#         # self.call(layers.Input(shape=(history_len, 1)))
#
#     def call(self, inputs, training=True):
#         x = self.dilation1(inputs)
#         x = self.dilation2(x)
#         x = self.slicer_layer(x)
#         x = self.dense(x)
#         return x
