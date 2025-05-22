import os.path as path
from typing import Any, TypeVar, Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from utils import rotation_z, rotation_xyz

MODEL_NAME = 'model_implant'
INPUT_DATASET = "bone_tooth_v2.pkl"
OUTPUT_DATASET = "bone_tooth_v2_sol.pkl"
INPUT_DATASET_ORIG = 'bone_tooth_v2_orig.pkl'
OUTPUT_DATASET_ORIG = 'bon_tooth_v2_sol_orig.pkl'
TRANSF_DATASET = 'bone_tooth_v2_sol_trans.pkl'

# Dental arc = the ideal line connecting the tooth
# LLD = Labio Lingual Diameter , the orthogonal plane wrt the dental arc
# MDD = Mesio Distal Diameter , the parallel plane wrt the dental arc

# For the bones two different levels : at crown level (CEJ) and 2mm below

Bone_lld_mean = LLD1_mean = 0
Bone_lld_std = LLD1_std = 0

Bone_mdd_mean = MDD1_mean = 0
Bone_mdd_std = MDD1_std = 0

Bone_lld_cej_mean = LLD_mean = 0
Bone_lld_cej_std = LLD_std = 0

Bone_mdd_cej_mean = MDD_mean = 0
Bone_mdd_cej_std = MDD_std = 0

Bone_h_mean = Root_h_mean = TL_mean = 0
Bone_h_std = Root_h_std = TL_std = 0

Tooth_ldd_mean = Coronal_ldd_mean = LDDc_mean = 0
Tooth_ldd_std = Coronal_ldd_std = LDDc_std = 0

Tooth_mdd_mean = Coronal_mdd_mean = MDDc_mean = 0
Tooth_mdd_std = Coronal_mdd_std = MDDc_std = 0

Tooth_h_mean = Coronal_h_mean = CL_mean = 0
Tooth_h_std = Coronal_h_std = CL_std = 0


def read_dental() -> pd.DataFrame:
    ds = pd.read_csv(path.join(path.dirname(__file__), "dental.csv"), index_col=["tooth", "mes"],
                     decimal=",")
    return ds


class ToothFactory:
    def __init__(self, d_margin=2, b_margin=2, c_min_perc: float = -1.5, c_max_perc: float = -0.4):
        self.d_margin = d_margin
        self.b_margin = b_margin
        self.c_min_perc = c_min_perc
        self.c_max_perc = c_max_perc
        self.dental_data = read_dental()

    def generate_data_2d(self, count=1000):
        data = self.generate_data(count)
        data2 = data[data[:, :, 2] > 0].reshape((-1, 8, 3))
        data2 = np.delete(data2, 2, axis=-1).reshape((-1, 8, 2))
        return data2

    def generate_data(self, count=1000, tooth='molar') -> np.ndarray:
        """
        Generates count teeth of type tooth. The result is a sequence of 3d points.
        The first 8 points are relative to the tooth's root, reduced by the margins;
        the next 8 points are relative to the tooth's crown.
        The reference frame is oriented like this:
        - x-axis -> labio lingual axis
        - y-axis -> vertical axis
        - z-axis -> medio distal axis

        :param count: number of teeth to create
        :param tooth: the tooth type
        :return:
        """

        tooth_data = self.dental_data.loc[tooth, :]

        def gen_data(type: str) -> np.ndarray:
            dist = tooth_data[type]
            (mean, dev, mn, mx) = dist[["mean", "std", "min", "max"]]
            return np.maximum(np.minimum(np.random.normal(loc=mean, scale=dev, size=count), mx), mn)

        # Generate roots' heights and diameters
        def gen_cube(l_name: str, lld_name: str, mdd_name: str, dm=self.d_margin,
                     bm=self.b_margin) -> np.ndarray:
            rl = gen_data(l_name)
            lld1 = gen_data(lld_name) - dm
            mdd1 = gen_data(mdd_name) - dm

            (x1, x2) = (-lld1 / 2, lld1 / 2)
            (y1, y2) = (-rl + bm, np.zeros(count))
            (z1, z2) = (-mdd1 / 2, mdd1 / 2)

            cube = [
                [x1, y1, z1],
                [x2, y1, z1],
                [x2, y2, z1],
                [x1, y2, z1],
                [x1, y1, z2],
                [x2, y1, z2],
                [x2, y2, z2],
                [x1, y2, z2],
            ]

            return np.transpose(np.array(cube), (2, 0, 1))

        n = TypeVar("n", bound=int)

        def random_rotate(cube: np.ndarray[(n, 8, 3), Any],
                          h: np.ndarray[(n, 1), Any],
                          a: np.ndarray[(n, 1), Any]) -> np.ndarray[(n, 8, 3), Any]:
            """
            Rotate the cube by a random angle and around a random center on the y axis between
            c_min_perc and c_max_perc of h. The maximum angle depends on the a (which is the root
            width) and the ray is the hypothenusis of the rectangle angle ...
            :param cube:
            :param h:
            :param a:
            :return:
            """
            num = cube.shape[0]
            c_min = self.c_min_perc * h
            c_max = self.c_max_perc * h

            c = (c_max - c_min) * np.random.rand(num) + c_min

            b = c

            alpha_max = np.arctan2(a, -b)

            alpha = np.random.random() * alpha_max

            r = np.sqrt(a ** 2 + b ** 2)

            t = np.expand_dims([0, 1, 0] * np.expand_dims(r, -1), 1)

            t0 = cube + t

            rt = rotation_z(alpha)

            t1 = np.matmul(t0, rt)

            t = np.expand_dims([0, 1, 0] * np.expand_dims(c, -1), 1)

            t2 = t1 + t

            return t2

        root = gen_cube('rl', 'lld1', 'mdd1')
        crown = gen_cube('cl', 'lldc', 'mddc', bm=0, dm=0) * [1, -1, 1]

        h = -root[:, 0, 1]
        a = -root[:, 0, 0] + self.d_margin

        crown = random_rotate(crown, h, a)

        return np.concatenate([root, crown], axis=1)


def convert_sol(sol: tf.Tensor) -> np.ndarray:
    [ox, oy, c, s] = sol.numpy()[0]
    alpha = np.arctan2(s, c)
    return np.array([alpha, 0, alpha, ox, oy])


Points3d = np.ndarray[Literal["n", 3], np.float32]


def random_rotate_and_traslate(points: Points3d, t_max=4.0, max_theta=2 * np.pi) -> Points3d:
    thetas = (np.random.random((1, 3)) * 2 - 1) * max_theta
    rots = rotation_xyz(thetas)
    tras = np.random.random((1, 3)) * t_max - t_max / 2

    x = points
    x = np.matmul(x, rots) + tras
    return x
