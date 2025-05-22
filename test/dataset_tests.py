import unittest

import dataset_reader as dsr


class DataSetTestCase(unittest.TestCase):
    def test_reading(self):
        data = list(dsr.read_data())

        self.assertEqual(len(data), 105)

    def test_write(self):
        data = dsr.read_data()
        d1, d2 = dsr.save_to_pickle(data)
        print(d1)
        print(d2)

    def test_prepare_v3(self):
        inp, out = dsr.prepare_data_v3()
        print(inp.shape)
        print(out.shape)

    def test_prepare_v4(self):
        inp, out, perms = dsr.prepare_data_v4()
        print(inp.shape)
        print(out.shape)
        print(perms[0:10])


if __name__ == '__main__':
    unittest.main()
