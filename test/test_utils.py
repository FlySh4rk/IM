import unittest

import numpy as np
import tensorflow as tf
import keras
from utils import CrossProductLayer, PermLayer


class TestCrossProductLayer(unittest.TestCase):
    def test_dimension(self):
        k = CrossProductLayer(with_squares=False)
        x = keras.layers.Input(shape=16)
        y = k(x)
        m = keras.models.Model(x, y)
        m.summary()

        self.assertEqual((None, 136), m.output_shape, )

    def test_dimension2(self):
        k = CrossProductLayer(with_squares=True)
        x = keras.layers.Input(shape=16)
        y = k(x)
        m = keras.models.Model(x, y)
        m.summary()

        self.assertEqual((None, 152), m.output_shape, )

    def test_working2(self):
        k = CrossProductLayer(with_squares=True, initializer=tf.constant_initializer(value=1))
        x = keras.layers.Input(shape=4)
        y = k(x)
        m = keras.models.Model(x, y)
        inp = tf.constant(shape=(2, 4), value=np.arange(8))
        res = m(inp)
        print(np.asarray(res.numpy()).tolist())
        self.assertTrue(np.all(
            [[0.0, 1.0, 4.0, 9.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.5, 3.0],
             [16.0, 25.0, 36.0, 49.0, 4.0, 5.0, 6.0, 7.0, 10.0, 12.0, 14.0, 15.0, 17.5, 21.0]]
            == res))

    def test_working3(self):
        k = CrossProductLayer(with_squares=True, order=3,
                              initializer=tf.constant_initializer(value=1))
        x = keras.layers.Input(shape=4)
        y = k(x)
        m = keras.models.Model(x, y)
        m.summary()
        inp = tf.constant(shape=(2, 4), value=np.arange(8))
        res = m(inp)
        print(np.asarray(res.numpy()).tolist())
        self.assertTrue(np.all(
            [[0.0, 1.0, 4.0, 9.0, 0.0, 0.25, 2.0, 6.75, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.5,
              3.0, 0.0, 0.0, 0.0, 1.0],
             [16.0, 25.0, 36.0, 49.0, 16.0, 31.249998092651367, 54.0, 85.75, 4.0, 5.0, 6.0, 7.0,
              10.0, 12.0, 14.0, 15.0, 17.5, 21.0, 20.0, 23.33333396911621, 28.0, 35.0]]
            == res))

    def test_working(self):
        k = CrossProductLayer(with_squares=False, initializer=tf.constant_initializer(value=1))
        x = keras.layers.Input(shape=16)
        y = k(x)
        m = keras.models.Model(x, y)
        inp = tf.constant(shape=(2, 16), value=np.arange(32))
        res = m(inp)
        self.assertTrue(np.all([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                                 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 6.0, 7.5, 9.0, 10.5, 12.0,
                                 13.5, 15.0, 16.5, 18.0, 19.5, 21.0, 22.5, 10.0, 12.0, 14.0, 16.0,
                                 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 15.0, 17.5, 20.0, 22.5,
                                 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 21.0, 24.0, 27.0, 30.0, 33.0,
                                 36.0, 39.0, 42.0, 45.0, 28.0, 31.5, 35.0, 38.5, 42.0, 45.5, 49.0,
                                 52.5, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 45.0, 49.5, 54.0,
                                 58.5, 63.0, 67.5, 55.0, 60.0, 65.0, 70.0, 75.0, 66.0, 71.5, 77.0,
                                 82.5, 78.0, 84.0, 90.0, 91.0, 97.5, 105.0],
                                [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
                                 27.0, 28.0, 29.0, 30.0, 31.0, 136.0, 144.0, 152.0, 160.0, 168.0,
                                 176.0, 184.0, 192.0, 200.0, 208.0, 216.0, 224.0, 232.0, 240.0,
                                 248.0, 153.0, 161.5, 170.0, 178.5, 187.0, 195.5, 204.0, 212.5,
                                 221.0, 229.5, 238.0, 246.5, 255.0, 263.5, 171.0, 180.0, 189.0,
                                 198.0, 207.0, 216.0, 225.0, 234.0, 243.0, 252.0, 261.0, 270.0,
                                 279.0, 190.0, 199.5, 209.0, 218.5, 228.0, 237.5, 247.0, 256.5,
                                 266.0, 275.5, 285.0, 294.5, 210.0, 220.0, 230.0, 240.0, 250.0,
                                 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 231.0, 241.5, 252.0,
                                 262.5, 273.0, 283.5, 294.0, 304.5, 315.0, 325.5, 253.0, 264.0,
                                 275.0, 286.0, 297.0, 308.0, 319.0, 330.0, 341.0, 276.0, 287.5,
                                 299.0, 310.5, 322.0, 333.5, 345.0, 356.5, 300.0, 312.0, 324.0,
                                 336.0, 348.0, 360.0, 372.0, 325.0, 337.5, 350.0, 362.5, 375.0,
                                 387.5, 351.0, 364.0, 377.0, 390.0, 403.0, 378.0, 391.5, 405.0,
                                 418.5, 406.0, 420.0, 434.0, 435.0, 449.5, 465.0]] == res.numpy()))


class TestPermLayer(unittest.TestCase):
    def test_dimension(self):
        k = PermLayer(num_permutation=10, flatten=False, groups=[(0, 4), (4, 8)])
        x = keras.layers.Input(shape=16)
        y = k(x)
        m = keras.models.Model(x, y)
        m.summary()
        self.assertEqual((None, 10, 16), m.output_shape)
        print(m.output_shape)
        i = tf.convert_to_tensor([list(range(16)), list(range(16))])

        r = m(i)
        self.assertEqual((2, 10, 16), r.shape)
        print(r.shape)
        print(r)
        print(tf.reshape(r, shape=(-1, 16)))

    def test_dimension_flat(self):
        k = PermLayer(num_permutation=10, flatten=True, groups=[(0, 4), (4, 8)])
        x = keras.layers.Input(shape=16)
        y = k(x)
        m = keras.models.Model(x, y)
        m.summary()
        self.assertEqual((None, 16), m.output_shape)
        print(m.output_shape)
        i = tf.convert_to_tensor([list(range(16)), list(range(16))])
        r = m(i)
        self.assertEqual((20, 16), r.shape)
        print(r.shape)
        print(r)
        print(tf.reshape(r, shape=(-1, 16)))

    def test_read_dental(self):
        from second_util import read_dental

        print(read_dental())

    def test_factory(self):
        from second_util import ToothFactory

        tooth_factory = ToothFactory()
        data = tooth_factory.generate_data_2d(count=1)
        print(data)

    def test_randrottrans(self):
        from second_util import random_rotate_and_traslate, ToothFactory
        tooth_factory = ToothFactory()
        data = tooth_factory.generate_data(count=1)
        data = random_rotate_and_traslate(data[0])
        print(data)


if __name__ == '__main__':
    unittest.main()
