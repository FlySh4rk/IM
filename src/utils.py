import math
import os.path as path
from itertools import combinations
from typing import Optional, Callable, List, Tuple, Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import Model
import keras

DATADIR = path.join(path.dirname(path.dirname(__file__)), "data")


def datadir(*p: str) -> str:
    return path.join(DATADIR, *p)


def add_cross_prods(r: np.ndarray) -> np.ndarray:
    prods = np.array(
        [r[comb[0]] * r[comb[1]] for comb in combinations(range(r.shape[0]), 2)])
    return np.append(r, prods)


def add_cross_prods_and_squares(r: np.ndarray) -> np.ndarray:
    x = np.append(r, [1])
    prods = np.array(
        [x[comb[0]] * x[comb[1]] for comb in combinations(range(x.shape[0]), 2)])
    prods = np.append(prods, r ** 2)
    return prods


def apply_model(model: keras.Model, row: np.ndarray) -> np.ndarray:
    inp = add_cross_prods(row)
    return model(inp.reshape(1, -1)).numpy()


def rotation(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


n = TypeVar("n")


def rotation_z(theta: np.ndarray[(n,), Any]) -> np.ndarray[(n, 3, 3), Any]:
    y = np.array([np.cos(theta), np.sin(theta)]).T
    x = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, -1, 0], [1, 0, 0], [0, 0, 0]]])
    z = np.tensordot(y, x, axes=([1], [0]))
    return z + np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])


def rotation_y(theta: np.ndarray[(n,), Any]) -> np.ndarray[(n, 3, 3), Any]:
    y = np.array([np.cos(theta), np.sin(theta)]).T
    x = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 1]], [[0, 0, -1], [0, 0, 0], [1, 0, 0]]])
    z = np.tensordot(y, x, axes=([1], [0]))
    return z + np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])


def rotation_x(theta: np.ndarray[(n,), Any]) -> np.ndarray[(n, 3, 3), Any]:
    y = np.array([np.cos(theta), np.sin(theta)]).T
    x = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 0, 0], [0, 0, -1], [0, 1, 0]]])
    z = np.tensordot(y, x, axes=([1], [0]))
    return z + np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])


def rotation_xyz(theta: np.ndarray[(n, 3), Any]) -> np.ndarray[(n, 3, 3), Any]:
    return np.matmul(rotation_x(theta[:, 0]),
                     np.matmul(rotation_y(theta[:, 1]),
                               rotation_z(theta[:, 2])))


def rotate(points: np.ndarray, theta: float, center=None) -> np.ndarray:
    if center is None:
        center = np.array([0., 0.])
    else:
        center = np.array(center)
    rot = rotation(theta)
    points = points.reshape(-1, 2) - center
    points = np.dot(points, rot) + center
    return points.reshape(-1)


# fig1 = plt.figure()

# ax = fig1.add_axes([-2, -2, 4, 4])


def order(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    x = points - c
    t = np.arctan2(x[:, 0], x[:, 1])
    t = (t + 2 * np.pi) % (2 * np.pi)
    idxs = np.argsort(t)
    res = points[idxs]
    return res


def draw_square(ax: plt.Axes, row: np.ndarray, color="red"):
    points = row.reshape(-1, 2)
    points = order(points)
    ax.plot(points[:, 0], points[:, 1],
            color=color, linewidth=2)
    ax.plot(points[[3, 0], 0], points[[3, 0], 1],
            color=color, linewidth=2)


# display(order(prepared_input[1, 0:8].reshape(-1, 2)))
def draw_input(ax: plt.Axes, row: np.ndarray):
    draw_square(ax, row[0:8], color="green")
    draw_square(ax, row[8:16], color="black")


def draw_output(ax: plt.Axes, row: np.ndarray, color="gray"):
    points = row.reshape(-1, 2)
    ax.plot(points[:, 0], points[:, 1],
            color=color, linewidth=3)


def draw_row(model: Optional[keras.Model], ax: plt.Axes, row: np.ndarray,
             apply: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    if apply is None:
        # apply = lambda r: apply_model(model, r)
        def apply(r: np.ndarray) -> np.ndarray:
            return apply_model(model, r)
    out = apply(row)
    draw_input(ax, row)
    draw_output(ax, out)


def draw_single(model: keras.Model, row: np.ndarray, fig: plt.Figure = None) -> plt.Figure:
    if fig is None:
        fig = plt.figure()

    # ax = fig.add_axes()
    ax = fig.axes[0]
    draw_row(model, ax, row)
    return fig


def draw_samples(model: Optional[keras.Model], input_dataset: pd.DataFrame, num_samples=4, num_rows=2,
                 apply: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    idxs = np.random.randint(0, input_dataset.shape[0] + 1, size=num_samples * num_rows)
    fig, axs = plt.subplots(num_rows, num_samples, figsize=(10, 10))  # , figsize=(10, 2))
    for i in range(len(idxs)):
        r = i // num_samples
        c = i % num_samples
        a: plt.Axes = axs[r][c]
        a.set_title(f"Sample {idxs[i]} ({i + 1}/{len(idxs)}")
        # a.inset_axes(bounds=[-2, -2, 4, 6])
        draw_row(model, a, input_dataset.to_numpy()[idxs[i]], apply)

    plt.show(fig)


def split_input(inpts: np.ndarray, outs: np.ndarray, shuffle=True):
    indexes = np.arange(inpts.shape[0])
    if shuffle:
        np.random.shuffle(indexes)
    train_size = math.ceil(inpts.shape[0] * 9 / 10)
    train_indexes = indexes[0:train_size]
    test_indexes = indexes[train_size:-1]
    return (inpts[train_indexes], outs[train_indexes]), (inpts[test_indexes], outs[test_indexes])


class CrossProductLayer(keras.layers.Layer):
    def __init__(self, with_squares=True, order=2, initializer=tf.random_normal_initializer,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_squares = with_squares
        self.w = None
        self.order = order
        self.initializer = initializer

    def build(self, input_shape):
        num_inputs = input_shape[1]
        num_weights = 0
        if self.with_squares:
            num_weights += num_inputs * (self.order - 1)
        import math
        num_weights += np.sum([math.comb(num_inputs, k + 1) for k in range(self.order)])

        self.w = self.add_weight(shape=(num_weights,), dtype=tf.float32, trainable=True,
                                 initializer=self.initializer, name="xcross_weights")
        return super().build(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        a = []

        if self.with_squares:
            for o in range(1, self.order):
                y = (inputs ** (o + 1)) / (o ** 2)
                a.append(y)

        f = 1
        for o in range(1, self.order + 1):
            f = f * o
            x = [
                tf.reduce_prod(tf.gather(inputs, axis=-1, indices=comb), axis=-1, keepdims=True) / f
                for
                comb in
                combinations(range(inputs.shape[-1]), o)
            ]
            tx = tf.concat(x, axis=-1)
            a.append(tx)

        t = tf.concat(a, axis=-1)

        return t * self.w

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


# for each input generates random permutations of some set of indices
class PermLayer(keras.layers.Layer):
    def __init__(self, num_permutation: int, groups: List[Tuple[int, int]], flatten=True, **kwargs):
        super().__init__(**kwargs)
        self.num_permutation = num_permutation
        self.groups = groups
        self.flatten = flatten

    def build(self, input_shape):
        super().build(input_shape)
        # self.output_shape = (input_shape[0], input_shape[1], self.num_permutation)

    def call(self, inputs: tf.Tensor, *args, **kwargs):
        if tf.is_symbolic_tensor(inputs):
            res = tf.convert_to_tensor([inputs for _ in range(self.num_permutation)])
            res = tf.transpose(res, (1, 0, 2))
            if self.flatten:
                res = tf.reshape(res, (-1, inputs.shape[1]))
            return res

        def permutate(row):
            index = np.arange(row.shape[0] / 2, dtype=np.int32)
            for a, b in self.groups:
                index[a:b] = np.random.permutation(index[a:b])
            x = tf.reshape(row, shape=(-1, 2))
            x = tf.gather(x, axis=0, indices=index)
            x = tf.reshape(x, shape=(-1,))
            return x

        rows = [[permutate(row) for row in inputs] for _ in range(self.num_permutation)]
        rows = tf.convert_to_tensor(rows)
        rows = tf.transpose(rows, (1, 0, 2))
        if self.flatten:
            rows = tf.reshape(rows, (-1, inputs.shape[1]))
        return rows


Implant_m = 0
Implant_h = 10 + Implant_m
Implant_w = 4 + Implant_m * 2


def draw_solution(ax: plt.Axes, prob: np.ndarray, sol: Optional[np.ndarray] = None,
                  implant_h=Implant_h,
                  implant_w=Implant_w,
                  orig: Optional[np.ndarray] = None,
                  trans: Optional[np.ndarray] = None) -> None:
    [B0, B1, B2, B3] = prob[0].reshape(4, 2)
    [T0, T1, T2, T3] = prob[1].reshape(4, 2)

    draw_square(ax, prob[0], color="grey")
    draw_square(ax, prob[1], color="black")

    if orig is not None:
        [O0, O1, O2, O3] = orig[0].reshape(4, 2)
        x = np.concatenate([O0 + [-2, -2], O1 + [2, -2], O2 + [2, 0], O3 + [-2, 0]])
        if trans is not None:
            tx, ty, thetas, sig = trans
            rots = rotation(thetas)
            flip = np.array([[sig, 0], [0, 1]])
            x = x.reshape(4, 2)
            x = np.matmul(np.matmul(x, flip), rots) + [tx, ty]
        draw_square(ax, x, color="grey")

    if sol is not None:
        [alpha, typ, alpha_max, Ox, Oy] = sol

        O = sol[-2:]

        implant = np.array([[-implant_w / 2, 0], [implant_w / 2, 0], [implant_w / 2, -implant_h],
                            [-implant_w / 2, -implant_h]])

        # implant_int = np.array([[-implant_w / 2 + margin_w, 0], [implant_w / 2 - margin_w, 0],
        #                        [implant_w / 2 - margin_w, -implant_h + margin_b],
        #                        [-implant_w / 2 + margin_w, -implant_h + margin_b]])

        r = rotation(alpha)
        r_max = rotation(alpha_max)

        implant_fin = np.matmul(implant, r) + O
        # implant_fin_int = np.matmul(implant_int, r) + O
        implant_max = np.matmul(implant, r_max) + O
        # implant_max_int = np.matmul(implant_int, r_max) + O

        draw_square(ax, implant_fin, color="red")
        # draw_square(ax, implant_fin_int, color="red")
        draw_square(ax, implant_max, color="lightgrey")
        # draw_square(ax, implant_max_int, color="lightgrey")

        # Now draw the screw

        alpha_s = np.pi / 2 - (alpha + typ * np.pi / 36)
        L = 6

        ax.plot([O[0], O[0] + L * np.cos(alpha_s)], [O[1], O[1] + L * np.sin(alpha_s)],
                color="yellow")


def draw_solution_v4(ax: plt.Axes, ax1: plt.Axes,
                     prob: np.ndarray,
                     sol: Optional[np.ndarray] = None,
                     implant_h=Implant_h,
                     implant_w=Implant_w,
                     orig: Optional[np.ndarray] = None,
                     perm: List[int] = None,
                     trans: Optional[np.ndarray] = None) -> None:
    # Fix shape
    prob = prob.reshape(16, 3)
    sol = sol.reshape(7)

    # Fix perm
    inv = [perm.index(i) for i in range(16)]
    prob = prob[inv]

    o = sol[0:3]
    a1 = sol[3:5]
    a2 = sol[5:7]

    # ll (left) projection
    ll_x_index = 0
    ll_y_index = 1
    ll_f1_indexes = [0, 1, 2, 3]
    ll_f2_indexes = [8, 9, 10, 11]

    # md (front) projection
    md_x_index = 2
    md_y_index = 1
    md_f1_indexes = [0, 3, 4, 7]
    md_f2_indexes = [8, 11, 12, 15]

    def face(points: np.array, xi: int, yi: int, idx: List[int]):
        return points[idx][:, [xi, yi]]

    b_face = face(prob, ll_x_index, ll_y_index, ll_f1_indexes)
    t_face = face(prob, ll_x_index, ll_y_index, ll_f2_indexes)

    draw_square(ax, b_face, color="grey")
    draw_square(ax, t_face, color="black")

    md_b_face = face(prob, md_x_index, md_y_index, md_f1_indexes)
    md_t_face = face(prob, md_x_index, md_y_index, md_f2_indexes)

    draw_square(ax1, md_b_face, color="grey")
    draw_square(ax1, md_t_face, color="black")

    def draw_implant(ax, ax1, o, a1, a2, x_index, y_index, color="red"):
        # normalize sc
        a1 = a1 / np.sqrt(a1.T @ a1)
        a2 = a2 / np.sqrt(a2.T @ a2)
        implant = np.array([[-implant_w / 2, 0, -implant_w / 2],
                            [implant_w / 2, 0, -implant_w / 2],
                            [implant_w / 2, -implant_h, -implant_w / 2],
                            [-implant_w / 2, -implant_h, -implant_w / 2],
                            [-implant_w / 2, 0, +implant_w / 2],
                            [implant_w / 2, 0, +implant_w / 2],
                            [implant_w / 2, -implant_h, +implant_w / 2],
                            [-implant_w / 2, -implant_h, +implant_w / 2]])

        # implant_int = np.array([[-implant_w / 2 + margin_w, 0], [implant_w / 2 - margin_w, 0],
        #                        [implant_w / 2 - margin_w, -implant_h + margin_b],
        #                        [-implant_w / 2 + margin_w, -implant_h + margin_b]])

        r1 = np.array([[a2[0], a2[1], 0],
                       [-a2[1], a2[0], 0],
                       [0, 0, 1], ])

        r2 = np.array([[1, 0, 0],
                       [0, a1[0], -a1[1]],
                       [0, a1[1], a1[0]]])

        implant_fin = (implant @ r1 @ r2) + o

        # implant_fin = np.matmul(implant, r) + O
        # implant_fin_int = np.matmul(implant_int, r) + O
        # implant_max_int = np.matmul(implant_int, r_max) + O

        f = face(implant_fin, x_index, y_index, [0, 1, 2, 3])

        draw_square(ax, f, color=color)

        md_f = face(implant_fin, md_x_index, md_y_index, [0, 3, 4, 7])
        draw_square(ax1, md_f, color=color)
        # draw_square(ax, implant_fin_int, color="red")
        # draw_square(ax, implant_max_int, color="lightgrey")

        # Now draw the screw

        # alpha_s = np.pi / 2 - (alpha + typ * np.pi / 36)
        # L = 6

        # ax.plot([O[0], O[0] + L * np.cos(alpha_s)], [O[1], O[1] + L * np.sin(alpha_s)],
        #        color="yellow")

    if sol is not None:
        O = o[[ll_x_index, ll_y_index]]
        draw_implant(ax, ax1, o, a1, a2, ll_x_index, ll_y_index)

    if orig is not None:
        orig = orig.reshape(7)
        o2 = orig[0:3]
        b1 = orig[3:5]
        b2 = orig[5:7]
        draw_implant(ax, ax1, o2, b1, b2, ll_x_index, ll_y_index, color="grey")


# [[a, b], [c, d]] -> [[a, c], [b, d]]
if __name__ == "__main__":
    print(datadir("a", "b"))
