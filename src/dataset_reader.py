import json
import logging
import random
from os import path
from typing import Iterator, Tuple, List

import numpy as np
import pandas as pd

from utils import datadir

logger = logging.getLogger("Data reader")

PROD_DATA = path.join(path.dirname(path.dirname(__file__)), "prod_data",
                      "implantMaster-solution.json")
APPROVED_AUTHORS = [
    "Silvio Emanuelli",
    "Davide Asborno"
]


def read_data(file: str = PROD_DATA):
    with open(file) as f:
        content = json.load(f)

    for entry in content:
        id = entry["_id"]["$oid"]
        author = entry["createdBy"]["name"]
        if author not in APPROVED_AUTHORS:
            logging.debug(f"Skipping author {author}")
            continue
        if "solution" not in entry:
            logging.debug(f"Skipping case {id} because it has no solution")
            continue
        problem = [
            [p["x"], p["y"], p["z"]] for p in
            (entry["problem"]["centerTooth"]['root'] + entry["problem"]["centerTooth"]['crown'])
        ]
        solution = [
            entry["solution"]["point"]["x"],
            entry["solution"]["point"]["y"],
            entry["solution"]["point"]["z"],
            entry["solution"]["rotationX"],
            entry["solution"]["rotationZ"]
        ]

        yield id, problem, solution


def save_to_pickle(data: Iterator[Tuple[str, List, List]]):
    data2 = {
        id: [np.array(prob), np.array(sol)] for id, prob, sol in data
    }

    ids = list(data2.keys())

    problems = np.array([
        data2[i][0] for i in ids
    ])

    sols = np.array([
        data2[i][1] for i in ids
    ])

    input_ds = pd.DataFrame(data=problems.reshape(-1, 16 * 3),
                            index=pd.Index(ids, dtype=np.str_, name="num"))

    input_ds.to_pickle(datadir("prod_prob_data.pkl"))

    output_ds = pd.DataFrame(data=sols,
                             index=pd.Index(ids, dtype=np.str_, name="num"))

    output_ds.to_pickle(datadir("prod_sol_data.pkl"))

    return input_ds, output_ds


def prepare_data_v3():
    input_data = pd.read_pickle(datadir("prod_prob_data.pkl"))
    output_data = pd.read_pickle(datadir("prod_sol_data.pkl")).to_numpy()

    prepared_input = np.array([o for o, _ in shuffle_points(input_data.to_numpy())])
    prepared_output = np.repeat(output_data,
                                repeats=prepared_input.shape[0] / output_data.shape[0], axis=0)

    return prepared_input, prepared_output


def prepare_data_v4(angle_weight: float = 10, num=128) -> Tuple[
    np.ndarray, np.ndarray, List[List[int]]]:
    input_data = pd.read_pickle(datadir("prod_prob_data.pkl"))
    output_data = pd.read_pickle(datadir("prod_sol_data.pkl")).to_numpy()

    c1 = np.cos(output_data[:, 3]).reshape(-1, 1) * angle_weight
    s1 = np.sin(output_data[:, 3]).reshape(-1, 1) * angle_weight
    c2 = np.cos(output_data[:, 4]).reshape(-1, 1) * angle_weight
    s2 = np.sin(output_data[:, 4]).reshape(-1, 1) * angle_weight

    output_data = np.concatenate([output_data[:, 0:3], c1, s1, c2, s2], axis=1)

    inputs = []
    perms = []

    for inp, perm in shuffle_points(input_data.to_numpy(), num=num):
        inputs.append(inp)
        perms.append(perm)

    prepared_input = np.array(inputs)
    prepared_output = np.repeat(output_data,
                                repeats=int(prepared_input.shape[0] / output_data.shape[0]),
                                axis=0)

    return prepared_input, prepared_output, perms


def shuffle_points(s: np.ndarray, num=100) -> Iterator[Tuple[np.ndarray, List[int]]]:
    for r in s:
        for i in range(num):
            p1 = list(range(8))
            random.shuffle(p1)
            p2 = list(range(8, 16))
            random.shuffle(p2)
            idxs = p1 + p2
            x = np.reshape(r, (16, 3))
            x = x[idxs, :]
            x = np.reshape(x, -1)
            yield x, idxs


def prepare_input(orig: pd.DataFrame, num=100) -> Iterator[np.ndarray]:
    for row, _ in shuffle_points(orig.to_numpy(), num):
        yield row
