from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Model


def add_cross_prods(r: np.ndarray) -> np.ndarray:
    prods = np.array(
        [r[comb[0]] * r[comb[1]] for comb in combinations(range(r.shape[0]), 2)])
    return np.append(r, prods)


def apply_model(model: Model, row: np.ndarray) -> np.ndarray:
    inp = add_cross_prods(row)
    return model(inp.reshape(1, -1)).numpy()


def rotation(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


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
    t = (t+2*np.pi)%(2*np.pi)
    idxs = np.argsort(t)
    res = points[idxs]
    return res


def draw_square(ax: plt.Axes, row: np.ndarray, color="red"):
    points = row.reshape(-1, 2)
    points = order(points)
    ax.plot(points[:, 0], points[:, 1],
            color=color)
    ax.plot(points[[3, 0], 0], points[[3, 0], 1],
            color=color)


# display(order(prepared_input[1, 0:8].reshape(-1, 2)))
def draw_input(ax: plt.Axes, row: np.ndarray):
    draw_square(ax, row[0:8], color="green")
    draw_square(ax, row[8:16], color="red")


def draw_output(ax: plt.Axes, row: np.ndarray, color="gray"):
    points = row.reshape(-1, 2)
    ax.plot(points[:, 0], points[:, 1],
            color=color)


def draw_row(model: Model, ax: plt.Axes, row: np.ndarray):
    out = apply_model(model, row)
    draw_input(ax, row)
    draw_output(ax, out)


def draw_single(model: Model, row: np.ndarray, fig: plt.Figure = None) -> plt.Figure:
    if fig is None:
        fig = plt.figure()

    #ax = fig.add_axes()
    ax = fig.axes[0]
    draw_row(model, ax, row)
    return fig


def draw_samples(model: Model, input_dataset: pd.DataFrame, num_samples=4, num_rows=2):
    idxs = np.random.randint(0, input_dataset.shape[0] + 1, size=num_samples * num_rows)
    fig, axs = plt.subplots(num_rows, num_samples, figsize=(10, 10))  # , figsize=(10, 2))
    for i in range(len(idxs)):
        r = i // num_samples
        c = i % num_samples
        a: plt.Axes = axs[r][c]
        a.set_title(f"Sample {idxs[i]} ({i + 1}/{len(idxs)}")
        # a.inset_axes(bounds=[-2, -2, 4, 6])
        draw_row(model, a, input_dataset.to_numpy()[idxs[i]])

    plt.show(fig)
