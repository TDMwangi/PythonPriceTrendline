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