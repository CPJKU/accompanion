# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def melody_lead(pitch, velocity, lead=0.02):
    """
    Compute melody lead
    """
    return (np.exp((pitch - 127) / 127.) *
            np.exp(-(velocity - 127) / 127) * lead)


def friberg_sundberg_rit(len_c, r_w=0.5, r_q=2.0):
    """
    Compute ritard_curve
    """
    if len_c > 0:
        ritard_curve = (1 + (r_w ** r_q - 1) *
                        np.linspace(0, 1, len_c)) ** (1. / r_q)
        return 1.0 / ritard_curve

    else:
        return 1.0


if __name__ == '__main__':

    rit = friberg_sundberg_rit(10)

    plt.plot(rit)
    plt.show()
