#!/usr/bin/python
""" Provides utility functions used by all other local modules.

Uh. Some of these are written okay. Some of them are questionable at best.

Read at your own discretion. I've tried to keep the more useful functions
documented to the best of my ability.

There are some comments I've written upon reviewing my own code.
"""

# STDlib
import time

# 3rd party lib
from colorama import Fore, Back, Style, init
import numpy as np
import progressbar


# yeah... wtf? why not put this inside a global store or something urgh
EXPONENT = "^"
BRACKETS = "{}"

ACCENT = Fore.CYAN
ACCENT_BR = ACCENT + Style.BRIGHT
RESET = Style.RESET_ALL  # < laziness in the name of syntactic sugar at its very finest


def progress(func, times, message):
    """ Runs a function many times, displaying a progressbar to show progress.
    Will not newline upon completion; can use print_after_progress() to show text afterwards.

    Args:
        func: function to run multiple times. Must take no arguments.
        times: number of times to run.
        message: text to show.
    Returns:
        Time taken for each function run on average.
    """

    START = time.time()

    progress = progressbar.ProgressBar(maxval=times, widgets=[
        ACCENT + Style.BRIGHT,
        message % times,
        Style.RESET_ALL,
        progressbar.Percentage(),
        progressbar.Bar()
    ])
    progress.start()

    for i in range(times):
        progress.update(i)
        func()
    progress.finish(end="")

    END = time.time()

    return (END-START) / times


def print_after_progress(text: str = "", end="\n"):
    """ Print a string after a progressbar, deleting that progressbar entirely.

    Args:
        text: text to print. Defaults to "".
        end: final character to print with. Defaults to "\n".
            Similar to print("", end="\n")
    Returns:
        None
    """

    print("\r%s\033[K" % text, end=end)


def format_float_exponent(number, decimals=100, brackets="{}", exponent="^"):
    """Format a float as a number to a power of ten.

    Args:
        number: number to format
        decimals: number of decimals to show
        brackets: what bracket to use for the exponent i.e. 10^{2}.
            Should be a string with two characters i.e. "{}"
        exponent: symbol for the exponent. Defaults to "^".

    Returns:
        String with the formatted number.
    """

    formatted = np.format_float_scientific(
        number, precision=decimals, trim="."
    )

    # replace "e-a" -> "10^{a}"
    formatted = formatted.replace(
        "e", "*10" + exponent + brackets[0]
    ) + brackets[1]

    return formatted


def generate_report(NAME: str, ORIGINAL_DATA, XVALS, model, times=10):
    """
    Generate a report, including model accuracy, time taken to create,
    and time taken to predict a set of values.

    Args:
        NAME (str): name of the model, to be used in the report.
        ORIGINAL_DATA (array_like): original data to be used in comparisons.
        model (func): the function to be called.
            Should accept no params, use utils.wrapper(func, *args, **kwargs)

    Returns:
        Dict with keys: {
            "differencesquares" (int): sum of different squares (after rounding predicted data),
            "incorrectvalues" (array_like): incorrect values,
            "generation_time": time taken to generate a model,
            "prediction_time": time taken to predict a set of values with the model,
            "model": generated model data
        }
    """

    MODEL_DATA = model()
    GENERATE_MODEL_AVG_TIME = 0
    PREDICT_MODEL_AVG_TIME = 0

    accuracy = generate_accuracy(
        NAME, ORIGINAL_DATA, MODEL_DATA["model"](XVALS))

    GENERATE_MODEL_AVG_TIME = progress(
        model, times, NAME + ": Generating %d models")
    PREDICT_MODEL_AVG_TIME = progress(
        lambda: MODEL_DATA["model"](XVALS), times *
        100, NAME + ": Predicting %d models"
    )

    print_after_progress(end="")

    return {
        **accuracy,  # combine accuracy dict into the new dict
        "generation_time": GENERATE_MODEL_AVG_TIME,
        "prediction_time": PREDICT_MODEL_AVG_TIME,
        "model": MODEL_DATA
    }


def generate_accuracy(NAME, ORIGINAL_DATA, PREDICTED_DATA):
    """ Generate the 'accuracy' of a model. Uses two values: the sum of difference squares
    and the number of correct values.

    Args:
        NAME: name of the model
        ORIGINAL_DATA (array_like of ints): original dataset for comparison
        PREDICTED_DATA (array_like): predicted dataset for comparison

    Returns:
        Dict of structure: {
            "differencesquares" (int): sum of different squares (after rounding predicted data),
            "incorrectvalues" (array_like): incorrect values
        }
    """
    LSSUM = sumOfDifferenceSquares(ORIGINAL_DATA, np.around(PREDICTED_DATA))

    DIFF = np.subtract(np.around(PREDICTED_DATA), ORIGINAL_DATA)
    INACCURATE = DIFF[np.nonzero(DIFF)]

    return {
        "differencesquares": LSSUM,
        "incorrectvalues": INACCURATE
    }


def timestampToSeconds(timestamp: str):
    """xx:xx -> minutes past 17:00"""
    if not timestamp:
        return None
    try:
        hour, minute = map(int, timestamp.split(":"))
        return int((hour - 17) * 60 + minute)
    except:
        return timestamp


def sumOfDifferenceSquares(DATA_1, DATA_2):
    return np.sum(
        np.square(
            DATA_1 - DATA_2
        )
    )


def wrapper(func, *args, **kwargs):
    def f():
        return func(*args, **kwargs)
    return f
