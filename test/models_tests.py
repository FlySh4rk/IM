import unittest

import matplotlib.pyplot as plt
import tensorflow as tf

from dataset_reader import prepare_data_v3, prepare_data_v4
from models import setup_model_v3
from models import setup_model_v4
from utils import split_input, draw_solution_v4


class ModelsTestCase(unittest.TestCase):
    def test_setup_v3(self):
        model = setup_model_v3()
        model.summary()

    def test_train_v3(self):
        input_data, output_data = prepare_data_v3()
        (input_train, output_train), (input_test, output_test) = split_input(input_data,
                                                                             output_data)

        model = setup_model_v3()
        BATCH_SIZE = 128
        res = model.fit(x=input_train, y=output_train, batch_size=BATCH_SIZE, epochs=20,
                        validation_data=(input_test, output_test))
        print(res)

    def test_setup_v4(self):
        model = setup_model_v4()
        model.summary()

    def test_train_v4(self):
        input_data, output_data = prepare_data_v4()
        (input_train, output_train), (input_test, output_test) = split_input(input_data,
                                                                             output_data)

        model = setup_model_v4()
        model.summary()
        BATCH_SIZE = 128
        res = model.fit(x=input_train, y=output_train, batch_size=BATCH_SIZE, epochs=20,
                        validation_data=(input_test, output_test))
        print(res)

    def test_draw_solution_v4(self):
        input_data, output_data, perms = prepare_data_v4()
        model = setup_model_v4()
        model.summary()
        prob = input_data[10]
        perm = perms[10]
        orig = output_data[10]
        # inv = [perm.index(i) for i in range(16)]
        out = model(tf.convert_to_tensor([prob])).numpy()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        # ax = fig.add_axes((0, 0, 1, 1))
        # prob = prob.reshape(16, 3)[inv]
        draw_solution_v4(ax1, ax2, prob, out, orig=orig, perm=perm)
        fig.show()


if __name__ == '__main__':
    unittest.main()
