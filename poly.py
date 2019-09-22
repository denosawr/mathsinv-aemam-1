#!/usr/bin/env python

import sys

import numpy.polynomial.polynomial as poly
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore, Back, Style, init

import utils

np.seterr(all="ignore")

# global switch
DEBUG = False


def blank(*args, **kwargs):
    pass


_print = print


def gen_poly_model(XVALS, DATA_FLAT, degree: int = 15, debug: bool = DEBUG):
    """Polynomial fit model for data calculation. Uses np.polynomial.polyfit.

    Args:
        XVALS (array_like): one-dimensional array of x-values of data.
        DATA_FLAT (array_like): one-dimensional array of y-values of data.

            # Note: zip(XVALS, DATA_FLAT) = two-dimensional array in form (x, y) of data

        degree (int): the degree of the polynomial.

        debug (bool): show all print() statements.

    Returns:
        A dict with the following data:
        {
            "predictions" (np.array): map of the data's predictions to the x-values,
            "equation" (string): a textual approximation of the equation, potentially rounded,
            "coefficients" (np.array): exact coefficients of the equation,
            "degree" (int): the exact degree of the equation, identical to args["input"],

            "model" (func): takes input x-vals and returns predictions
        }

    Raises:
        None
    """

    # calculate polynomial fit

    print = _print if debug else blank

    COEFFS = poly.polyfit(XVALS, DATA_FLAT, degree)
    print("Coeffs:", COEFFS)

    terms = []
    for index, weight in enumerate(COEFFS):
        coefficient = utils.format_float_exponent(weight)

        terms.append(coefficient + "x" + utils.EXPONENT +
                     utils.BRACKETS[0] + str(index) + utils.BRACKETS[1])

    EQN = " + ".join(terms)
    print("Generated equation:", EQN)

    # generate model
    def model(XVALS):
        return poly.polyval(XVALS, COEFFS)

    # predict results
    PREDICTIONS = model(XVALS)

    # should be theoretically equal - let's check it!
    DIFF = np.subtract(np.around(PREDICTIONS), DATA_FLAT)
    print(DIFF)
    for index, value in enumerate(DIFF):
        if value:
            print("DIFF [%d]: %d" % (index, value))

    return {
        "predictions": PREDICTIONS,
        "equation": EQN,
        "coefficients": COEFFS,
        "degree": degree,
        "model": model
    }


def gen_opt_poly_model(XVALS, DATA_FLAT, maxDegree: int = 30, debug: bool = DEBUG):
    """Optimise a polynomial fit model, performing a polyfit up to maxDegree. Uses np.polynomial.polyfit.

    Args:
        XVALS (array_like): one-dimensional array of x-values of data.
        DATA_FLAT (array_like): one-dimensional array of y-values of data.

            # Note: zip(XVALS, DATA_FLAT) = two-dimensional array in form (x, y) of data

        maxDegree (int): the maximum search degree of the polynomial.
            Will search all degrees from range(1, maxDegree+1), and return the most polyfit with the
            least squares fit. Returns smallest possible matching degree.

        debug (bool): show all print() statements.

    Returns:
        A dict with the following data:
        {
            "predictions" (np.array): map of the data's predictions to the x-values,
            "equation" (string): a textual approximation of the equation, potentially rounded,
            "coefficients" (np.array): exact coefficients of the equation,
            "degree" (int): the exact degree of the equation, identical to args["input"],

            "model" (func): takes input x-vals and returns predictions
        }

    Raises:
        None
    """

    print = _print if debug else blank

    values = np.full(maxDegree, np.inf)
    MODELS = []

    bestDiff = np.inf
    optimal = None

    print("Looping through degrees...")
    for degree in range(1, maxDegree+1):
        model = gen_poly_model(XVALS, DATA_FLAT, degree=degree)

        diff = utils.sumOfDifferenceSquares(
            DATA_FLAT, np.around(model["predictions"]))
        if diff < bestDiff:
            print(degree, diff, "<", bestDiff)
            bestDiff = diff
            optimal = model

    print("Found optimal value at", optimal["degree"])
    return optimal
