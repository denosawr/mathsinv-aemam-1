#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore, Back, Style, init

import utils

# global switch
DEBUG = False


def blank(*args, **kwargs):
    pass


_print = print


def gen_trig_model(XVALS, DATA_FLAT, orig_increment=0.1, min_increment=0.00001, threshold=10, use_gradient_for_min: bool = False, debug: bool = DEBUG):
    """Trigonometric model for data calculation. Minimises square of differences to generate optimal curve.

    Args:
        XVALS (array_like): one-dimensional array of x-values of data.
        DATA_FLAT (array_like): one-dimensional array of y-values of data.

            # Note: zip(XVALS, DATA_FLAT) = two-dimensional array in form (x, y) of data

        orig_increment (float): Starting increment to use.
            Must be a multiple of ten of min_increment.
        min_increment (float): Minimum increment.
            Must be a factor of orig_increment with 10.

            # Note: (orig_increment / min_increment) % 10 == 0

        threshold (float/int): range of data to look from.
        use_gradient_for_min (bool): choose between using the minimum
            value of the graph or the point at which the derivative crosses zero.
            Changing this option ultimately has no effect on the output data, due to
            how similar they are; however, I suspect that False is faster.

        debug (bool): show all print() statements.

    Returns:
        A dict with the following data:
        {
            "predictions" (np.array): map of the data's predictions to the x-values,
            "equation" (string): a textual approximation of the equation, potentially rounded,
            "constants": { # corresponding to the graph of `a*cos(bx) + c`
                "amplitude": amplitude of graph (i.e. a),
                "period": period of graph (i.e. b),
                "vtrans": vertical translation constant (i.e. c)
            },

            "model" (func): takes input x-vals and returns predictions
        }

    Raises:
        ValueError: if `(orig_increment / min_increment) % 10 != 0`
    """

    print = _print if debug else blank

    if ((orig_increment/min_increment) % 10) != 0:
        raise ValueError("Ensure `(orig_increment / min_increment) % 10 == 0`")

    # calculate trig fit
    MIN = min(DATA_FLAT)
    MAX = max(DATA_FLAT)
    MID = (MIN + MAX) / 2
    AMPLITUDE = (MAX - MIN) / 2
    print("%d cos(ax) + %s, optimise for a" % (AMPLITUDE, MID))

    if use_gradient_for_min:
        print("Using gradient to find local min.")
        # find lowest point, examining the gradient
        GRA = np.gradient(DATA_FLAT)
        # print(GRA)

        # determine the point where this gradient goes from negatives to positives
        ORIG_START = np.argwhere(GRA < 0)[-1] - threshold
        ORIG_END = np.argwhere(GRA > 0)[0] + threshold
    else:
        print("Using minimum value to find local min.")
        MIN = np.where(DATA_FLAT == np.amin(DATA_FLAT))[0]
        ORIG_START = MIN[0] - threshold
        ORIG_END = MIN[-1] + threshold
    print("Discovered minimum value:", MIN)

    def recursedMin(start, end, increment):
        # calculate number of increments beforehand
        RECURSIONS = ((end - start) / increment) + 1

        # 65535 - arbritrary large value, no significant meaning
        values = np.full((int(RECURSIONS), 2), np.inf)

        for index, v in enumerate(np.arange(start, end, increment)):
            # v is the intended minimum point i.e. half the period
            # given the natural period of the sin graph is 2π:
            period = np.pi / v

            # translate the equation
            PREDICTED = AMPLITUDE * np.cos(period * XVALS) + MID

            # calculate Least Squares Sum
            LSSUM = utils.sumOfDifferenceSquares(DATA_FLAT, PREDICTED)
            values[index] = [LSSUM, period]

        # get index of lowest value in array
        lsqs = values[:, 0]
        minimumValue = np.where(lsqs == np.amin(lsqs))[0]

        if increment == min_increment:
            return values[minimumValue][0][1]
        else:
            # convert back to day
            minimumDay = start + minimumValue*increment

            print("Min at", minimumDay, "for increment", increment)
            return recursedMin(
                start=minimumDay - threshold*increment,
                end=minimumDay + threshold*increment,
                increment=increment/10
            )

    PERIOD = recursedMin(
        start=ORIG_START,
        end=ORIG_END,
        increment=orig_increment
    )
    print("Period determined to be %s after optimisation with increment %s" %
          (PERIOD, np.format_float_positional(min_increment)))

    # generate model
    def model(XVALS):
        return AMPLITUDE * np.cos(PERIOD * XVALS) + MID

    # create predictions
    PREDICTED = model(XVALS)

    return {
        "predictions": PREDICTED,
        "equation": "%s cos(%sx) + %s" % (
            utils.format_float_exponent(AMPLITUDE),
            utils.format_float_exponent(PERIOD),
            utils.format_float_exponent(MID)
        ),
        "constants": {
            "amplitude": AMPLITUDE,
            "period": PERIOD,
            "vtrans": MID
        },
        "model": model
    }
