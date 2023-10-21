import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_trend_line(support, pivot, slope, y):
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.array(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 otherwise
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support, pivot, init_slope, y):
    slope_unit = (y.max() - y.min() / len(y))

    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)