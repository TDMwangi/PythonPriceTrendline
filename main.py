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

    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)

        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5
        else:
            best_err = test_err
            best_slope = test_slope
            get_derivative = True

    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendline(data):
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # Optimize the slop for both trendlines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def high_low(high, low, close):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    line_points = coefs[0] * x + coefs[1]

    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


# Load data
data = pd.read_csv('bitcoin.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

# Resolve price scaling issues
data = np.log(data)
lookback = 30

support_slope = [np.nan] * len(data)
resist_slope = [np.nan] * len(data)

for i in range(lookback - 1, len(data)):
    candles = data.iloc[i - lookback + 1: i + 1]
    support_coefs, resist_coefs = high_low(
        candles['high'], candles['low'], candles['close'])
