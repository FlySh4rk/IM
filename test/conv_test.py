import unittest

import tensorflow as tf

from vconv_layer import OriginalConv, CubeConv, CubeSampler, CrossConv, CrossConvVariadic2, \
    CrossConvV2


class ConvTestCase(unittest.TestCase):
    def test_conv_orig(self):
        conv = OriginalConv()
        v = tf.convert_to_tensor([[[1, 1, 1],
                                   [2, 2, 2],
                                   [3, 3, 3]],
                                  [[10, 10, 10],
                                   [20, 20, 20],
                                   [30, 30, 30]]], dtype=tf.float32)

        i = tf.keras.layers.Input(shape=(3, 3), dtype=tf.float32)
        o = conv(i)

        m = tf.keras.Model(i, o)

        z = m(v)
        print(m.summary())
        print(z)

    def test_conv_cross(self):
        conv = CrossConv(num_convs=1)
        v = tf.convert_to_tensor([[[1, 1, 1],
                                   [2, 2, 2],
                                   [3, 3, 3]],
                                  [[10, 10, 10],
                                   [20, 20, 20],
                                   [30, 30, 30]]], dtype=tf.float32)

        i = tf.keras.layers.Input(shape=(3, 3), dtype=tf.float32)
        o = conv(i)

        m = tf.keras.Model(i, o)

        z = m(v)
        print(m.summary())
        print(z)

    def test_conv_cross_v2(self):
        conv = CrossConvV2(num_convs=1)
        v = tf.convert_to_tensor([[[1, 1, 1, 2, 2],
                                   [2, 2, 2, 2, 2],
                                   [3, 3, 3, 2, 2]],
                                  [[10, 10, 10, 2, 2],
                                   [20, 20, 20, 2, 2],
                                   [30, 30, 30, 2, 2]]], dtype=tf.float32)

        i = tf.keras.layers.Input(shape=(3, 5), dtype=tf.float32)
        o = conv(i)

        m = tf.keras.Model(i, o)

        z = m(v)
        print(m.summary())
        print(z)

    def test_conv_cross_with_features(self):
        conv = CrossConv(num_convs=1, out_feats=1)
        v = tf.convert_to_tensor([[[1, 1, 1, 1],
                                   [2, 2, 2, 1],
                                   ],
                                  [[10, 10, 10, 1],
                                   [20, 20, 20, 1],
                                   ]], dtype=tf.float32)

        i = tf.keras.layers.Input(shape=(2, 4), dtype=tf.float32)
        o = conv(i)

        m = tf.keras.Model(i, o)

        z = m(v)
        print(m.summary())
        print(z)

    def test_cube_sampler(self):
        sampler = CubeSampler()
        s = sampler(tf.convert_to_tensor([[1], [2]]))
        print(s)

    def test_conv_cube(self):
        conv = CubeConv(density=2, num_convs=3, num_weighters=1)
        v = tf.convert_to_tensor([[[1, 1, 1],
                                   [2, 2, 2],
                                   [3, 3, 3]],
                                  [[10, 10, 10],
                                   [20, 20, 20],
                                   [30, 30, 30]]], dtype=tf.float32)

        i = tf.keras.layers.Input(shape=(3, 3), dtype=tf.float32)
        o = conv(i)

        m = tf.keras.Model(i, o)

        z = m(v)
        print(m.summary())
        print(z)

    def test_conv_cube_simple(self):
        conv = CubeConv(density=2, num_convs=1, num_weighters=1, sampler_dim=2, input_dim=2)
        v = tf.convert_to_tensor([[[0, 0, 0]]], dtype=tf.float32)

        i = tf.keras.layers.Input(shape=(1, 3), dtype=tf.float32)
        o = conv(i)

        m = tf.keras.Model(i, o)

        z = m(v)
        print(m.summary())
        print(z)

    def test_conv_var2(self):
        conv = CrossConvVariadic2(num_convs=1)
        v = tf.convert_to_tensor([[[0, 0, 0]]], dtype=tf.float32)

        i = tf.keras.layers.Input(shape=(1, 3), dtype=tf.float32)
        o = conv(i)

        m = tf.keras.Model(i, o)

        z = m(v)
        print(m.summary())
        print(z)


if __name__ == '__main__':
    unittest.main()
