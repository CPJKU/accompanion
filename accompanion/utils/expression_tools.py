# -*- coding: utf-8 -*-
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def melody_lead(
    pitch: Union[np.ndarray, int],
    velocity: int,
    lead: float = 0.02,
) -> Union[np.ndarray, float]:
    """
    Compute melody lead
    """
    return np.exp((pitch - 127) / 127.0) * np.exp(-(velocity - 127) / 127) * lead


def friberg_sundberg_rit(
    len_c: int,
    r_w: float = 0.5,
    r_q: float = 2.0,
) -> Union[np.ndarray, float]:
    """
    Compute ritard_curve
    """
    if len_c > 0:
        ritard_curve = (1 + (r_w**r_q - 1) * np.linspace(0, 1, len_c)) ** (1.0 / r_q)
        return 1.0 / ritard_curve

    else:
        return 1.0


if __name__ == "__main__":
    for rw in np.linspace(0.1, 1, 10):
        rit = friberg_sundberg_rit(10, rw, 2)

        plt.plot(rit, label=f"{rw:.2f}")

    plt.legend(loc="best")
    plt.show()
    plt.clf()
    plt.close()

    for rq in np.linspace(0.1, 3, 10):
        rit = friberg_sundberg_rit(10, 0.5, rq)

        plt.plot(rit, label=f"{rq:.2f}")

    plt.legend(loc="best")
    plt.show()
