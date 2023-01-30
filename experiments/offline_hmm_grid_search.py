import argparse
from multiprocessing import Pool
import tqdm


from itertools import product

import numpy as np

from offline_score_follower import alignment_experiment


def hyper_parameter_experiment(params):

    ioi_precision, scale, trans_par, trans_var, obs_var, init_var = params

    od = f"{args.out_dir}_ioip{ioi_precision:.4f}_scale{scale:.4f}_tp{trans_par:.4f}_tv{trans_var:.4f}_ov{obs_var:.4f}_iv{init_var:.4f}"

    config = dict(
        follower_type="hmm",
        ioi_precision=ioi_precision,
        gumbel_transition_matrix_scale=scale,
        init_bp=0.5,
        trans_par=trans_par,
        trans_var=trans_var,
        obs_var=obs_var,
        init_var=init_var,
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

    args = parser.parse_args()

    if args.solo is None:
        raise ValueError("No input performance given!")

    if args.reference is None:
        raise ValueError("No references given!")

    IOI_PRECISIONS = [0.1, 0.5, 1]
    SCALES = [1, 1.5, 2]
    TRANS_PARS = [1.5, 2]
    TRANS_VARS = [0.5, 1]
    OBS_VARS = [0.5, 1]
    INIT_VARS = [0.5, 1]
    # IOI_PRECISIONS = np.linspace(0.1, 2, 5)
    # SCALES = np.linspace(0.1, 2, 5)
    # TRANS_PARS = np.linspace(0.1, 2, 5)
    # TRANS_VARS = np.linspace(0.1, 2, 5)
    # OBS_VARS = np.linspace(0.1, 2, 5)
    # INIT_VARS = np.linspace(0.1, 2, 5)
    parameters = list(
        product(
            IOI_PRECISIONS,
            SCALES,
            TRANS_PARS,
            TRANS_VARS,
            OBS_VARS,
            INIT_VARS,
        )
    )

    with Pool(14) as p:
        r = list(
            tqdm.tqdm(
                p.imap(hyper_parameter_experiment, parameters),
                total=len(parameters),
            )
        )
