import os
import numpy as np
import argparse

from skopt import dump
from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize
from skopt import dummy_minimize

from hyperspace.benchmarks import StyblinksiTang


stybtang = StyblinksiTang(7)


def run(results_dir, n_calls=200, acq_optimizer="lbfgs"):
    bounds = np.tile((-5., 5.), (7, 1))
    optimizers = [("gp_minimize", gp_minimize),
                  ("forest_minimize", forest_minimize),
                  ("gbrt_minimize", gbrt_minimize),
                  ("dummy_minimize", dummy_minimize)]

    for name, optimizer in optimizers:
        print(name)
#        model_dir = os.path.join(results_dir, name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        if name == "gp_minimize":
            res = optimizer(
                stybtang, bounds, random_state=0, n_calls=n_calls,
                noise=1e-10, verbose=True, acq_optimizer=acq_optimizer,
                n_jobs=-1)
        elif name == "dummy_minimize":
            res = optimizer(
                stybtang, bounds, random_state=0, n_calls=n_calls)
        else:
            res = optimizer(
                stybtang, bounds, random_state=0, n_calls=n_calls)

        dump(res, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_calls', nargs="?", default=50, type=int, help="Number of function calls.")
    parser.add_argument(
        '--n_runs', nargs="?", default=1, type=int, help="Number of runs.")
    parser.add_argument(
        '--acq_optimizer', nargs="?", default="lbfgs", type=str,
        help="Acquistion optimizer.")
    parser.add_argument(
        '--results_dir', nargs="?", default='./results', type=str,
        help="Where to save results")
    args = parser.parse_args()

    run(args.results_dir, args.n_calls, args.acq_optimizer)
