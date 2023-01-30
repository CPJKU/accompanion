import argparse
from multiprocessing import Pool
import tqdm


from itertools import product

import numpy as np

from offline_score_follower import alignment_experiment


def hyper_parameter_experiment(params):

    window_size, step_size, start_window_size_p = params

    start_window_size = int(np.round(start_window_size_p * window_size))

    od = f"{args.out_dir}_ws{window_size:03d}_ss{step_size:02d}_st{start_window_size:03d}"
    config = dict(
        follower_type="oltw",
        window_size=window_size,
        step_size=step_size,
        start_window_size=start_window_size,
    )
    alignment_experiment(
        solo_perf_fn=args.solo,
        reference_fn=args.reference,
        score_follower_kwargs=config,
        out_dir=od,
        make_plots=False,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run an offline alignment experiment")

    parser.add_argument(
        "--solo",
        "-s",
        help="Input Solo performance (as a match file)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--reference",
        "-r",
        help="Reference (as a list of match files)",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--out-dir",
        "-o",
        help="Output directory to store the results",
        type=str,
        default=".",
    )

    # parser.add_argument(
    #     "--make_plots",
    #     "-p",
    #     help="Store piano roll alignment plot",
    #     action="store_true",
    #     default=False,
    # )

    args = parser.parse_args()

    if args.solo is None:
        raise ValueError("No input performance given!")

    if args.reference is None:
        raise ValueError("No references given!")

    WINDOW_SIZES = [50, 100, 150, 200]
    STEP_SIZES = [1, 2, 5, 10, 20]
    START_WINDOW_SIZES = [0.5, 1]

    parameters = list(product(WINDOW_SIZES, STEP_SIZES, START_WINDOW_SIZES))

    with Pool(8) as p:
        r = list(
            tqdm.tqdm(
                p.imap(hyper_parameter_experiment, parameters),
                total=len(parameters),
            )
        )
