#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore, Back, Style, init

import utils
from poly import gen_poly_model, gen_opt_poly_model
from trig import gen_trig_model

# global switch
DEBUG = False


def blank(*args, **kwargs):
    pass


_print = print


def gen_combined_model(XVALS, DATA_FLAT,
                       # params for trig model
                       orig_increment=0.1, min_increment=0.00001, threshold=10, use_gradient_for_min: bool = False,
                       maxDegree: int = 30,  # params for polyfit
                       debug: bool = DEBUG):
    """Combined model for data calculation, using both a trig and a polyfit. Minimises square of differences to generate optimal curve.
    First tries to create optimal trigonometric curve, then polyfits the remainder.

    Args:
        XVALS (array_like): one-dimensional array of x-values of data.
        DATA_FLAT (array_like): one-dimensional array of y-values of data.

            # Note: zip(XVALS, DATA_FLAT) = two-dimensional array in form (x, y) of data

        # For the following, consult the documentation for gen_trig_model
        orig_increment (float)
        min_increment (float)
        threshold (float/int)
        use_gradient_for_min (bool)

        # For the following, consult the documentation for gen_poly_model
        maxDegree (int)

        debug (bool): show all print() statements.

    Returns:
        A dict with the following data:
        {
            "predictions" (np.array): map of the data's predictions to the x-values,
            "equation" (string): a textual approximation of the equation, potentially rounded,

            # applies to the trig section of the equation
            "trig": {
                "equation" (string): a textual approximation of the trigonometric part of the equation
                "constants": { # corresponding to the graph of `a*cos(bx) + c`
                    "amplitude": amplitude of graph (i.e. a),
                    "period": period of graph (i.e. b),
                    "vtrans": vertical translation constant (i.e. c)
                },
            },

            # applies to the poly section of the equation
            "poly": {
                "equation" (string): a textual approximation of the polyfit of the equation
                "coefficients" (np.array): exact coefficients of the equation,
                "degree" (int): the exact degree of the equation, identical to args["input"],
            },

            "model" (func): takes input x-vals and returns predictions
        }

    Raises:
        ValueError: if `(orig_increment / min_increment) % 10 != 0`
    """

    print = _print if debug else blank

    if ((orig_increment / min_increment) % 10):
        raise ValueError("Ensure `(orig_increment / min_increment) % 10 == 0`")

    # first generate a trig model for the half section
    START_CURVE_VALS = XVALS[:194]
    START_CURVE_DATA = DATA_FLAT[:194]

    TRIG_MODEL = gen_trig_model(START_CURVE_VALS, START_CURVE_DATA, orig_increment=orig_increment, min_increment=min_increment,
                                threshold=threshold, use_gradient_for_min=use_gradient_for_min, debug=debug)

    # predict this trig model for all xvals
    TRIG_PREDICTIONS = TRIG_MODEL["model"](XVALS)

    POLY_MODEL = gen_opt_poly_model(
        XVALS, DATA_FLAT-TRIG_PREDICTIONS, maxDegree=maxDegree, debug=debug)

    PREDICTIONS = TRIG_PREDICTIONS + POLY_MODEL["predictions"]

    def model(XVALS):
        return TRIG_MODEL["model"](XVALS) + POLY_MODEL["model"](XVALS)

    return {
        "predictions": PREDICTIONS,
        "equation": TRIG_MODEL["equation"] + " + " + POLY_MODEL["equation"],
        "trig": {
            "equation": TRIG_MODEL["equation"],
            "constants": TRIG_MODEL["constants"]
        },
        "poly": {
            "equation": POLY_MODEL["equation"],
            "coefficients": POLY_MODEL["coefficients"],
            "degree": POLY_MODEL["degree"]
        },
        "model": model
    }
