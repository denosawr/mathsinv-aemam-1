#!/usr/bin/env python
"""Main file of investigation.

I would like to apologise in advance for anyone who tries to read this file. It is written horribly.
I promise that my code looks better than this usually.

Throughout this entire project, my switching between camelCase, snake_case and ALL_CAPS is _very_ questionable.
My eyes are bleeding.

Please forgive me.
"""

__author__ = "Ollie Cheng"


# STD lib
import argparse
import json
import sys
import warnings

# 3rd paty packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style, init

# Local packages
import utils  # must be imported before other local
from poly import gen_poly_model, gen_opt_poly_model
from trig import gen_trig_model
from combined import gen_combined_model

# Disable np rankwarning
warnings.filterwarnings('ignore')

# colorama
init(autoreset=True)


# parse args
parser = argparse.ArgumentParser(
    description="Generate multiple models to fit data and then test them.")
parser.add_argument("--infile", default="sunsets.csv", type=str,
                    help="File to read data from. Must be CSV in format provided. (default: subsets.csv)")
parser.add_argument("--outfile", default="results.json", type=str,
                    help="File to output data to. Leave blank to not output. (default: results.json)")
parser.add_argument("--graph", default=True, type=bool,
                    help="Generate a graph of data. (default: True)")
parser.add_argument("--table", default=True, type=bool,
                    help="Generate a table of data. (default: True)")
parser.add_argument("--tries", default=200, type=int,
                    help="Number of times to generate model. Model will be predicted args.tries*100 times. (default: 200)")


args = parser.parse_args()


def main():
    # FETCH & PARSE DATA
    SUNSET_TIMES = []
    np.set_printoptions(suppress=True)

    # DATA_DA = pandas.DataArray representation of data
    DATA_DA = pd.read_csv(args.infile,
                          #  skiprows=1,
                          usecols=range(2, 14),
                          converters=dict(((x, utils.timestampToSeconds)
                                           for x in range(0, 14)))
                          )

    # Transform DA from rows/cols to cols/rows
    DATA_DA = DATA_DA.T

    # flatten 2d array to all be in the same array
    DATA_FLAT = np.concatenate(tuple(
        (x[~np.isnan(x)] for x in DATA_DA.to_numpy())
    ))
    XVALS = np.arange(1, 366)

    data = {}

    print("Beginning calculations... hold on! (%d generations, %d predictions)" %
          (args.tries, args.tries*100))

    POLYOPT_DATA = utils.generate_report("Opt poly", DATA_FLAT, XVALS, utils.wrapper(
        gen_opt_poly_model,
        XVALS,
        DATA_FLAT,
        maxDegree=50,
    ), times=args.tries)

    POLYFIX_DEGREE = 15
    POLYFIX_DATA = utils.generate_report("Fixed poly (δ%d)" % POLYFIX_DEGREE, DATA_FLAT, XVALS, utils.wrapper(
        gen_poly_model,
        XVALS,
        DATA_FLAT,
        degree=POLYFIX_DEGREE
    ), times=args.tries)

    TRIG_DATA_1 = utils.generate_report("Trig1", DATA_FLAT, XVALS, utils.wrapper(
        gen_trig_model,
        XVALS,
        DATA_FLAT
    ), times=args.tries)

    TRIG_DATA_2 = utils.generate_report("Trig2", DATA_FLAT, XVALS, utils.wrapper(
        gen_trig_model,
        XVALS[:180],
        DATA_FLAT[:180]
    ), times=args.tries)
    TRIG_DATA_2["model"]["predictions"] = TRIG_DATA_2["model"]["model"](XVALS)

    COMBINED_DATA = utils.generate_report("Combined", DATA_FLAT, XVALS, utils.wrapper(
        gen_combined_model,
        XVALS,
        DATA_FLAT,
        maxDegree=50,
    ), times=args.tries)

    data["Opt poly (δ%d)" % POLYOPT_DATA["model"]["degree"]] = POLYOPT_DATA
    data["Fix poly (δ%d)" % POLYFIX_DATA["model"]["degree"]] = POLYFIX_DATA
    data["Cos1"] = TRIG_DATA_1
    data["Cos2"] = TRIG_DATA_2
    data["Combined (δ%d)" % COMBINED_DATA["model"]["poly"]
         ["degree"]] = COMBINED_DATA

    TABLE_ROW = "%s{:>22}%s | " + " | ".join(("%s{:^14}%s",)*len(data.keys()))
    TRANSLATION_DICT = {
        "differencesquares": ("Sum of difference^2", int),
        "incorrectvalues": ("# of incorrect values", len),
        "generation_time": ("Generation time (ms)", lambda x: "{:.4f}".format(x*1000)),
        "prediction_time": ("Prediction time (ms)", lambda x: "{:.4f}".format(x*1000))
    }

    print("Calculations complete; final data:\n\n")

    print(TABLE_ROW.format("", *(k for k in data.keys())) %
          ((utils.ACCENT_BR, utils.RESET)*(len(data.keys())+1)))
    for key, translation in TRANSLATION_DICT.items():
        values = [value[key] for value in data.values()]

        # filter passed
        formatted = TABLE_ROW.format(
            translation[0], *map(translation[1], values)
        ) % (utils.ACCENT_BR, utils.RESET, *("",)*(2*len(data.keys())))

        print(formatted)

    # graph results

    COLOURS = ["firebrick", "royalblue",
               "darkorchid", "darkslategrey", "forestgreen"]

    for i, (key, value) in enumerate(data.items()):
        plt.plot(XVALS, value["model"]["predictions"],
                 color=COLOURS[i], label=key)

    plt.plot(XVALS, DATA_FLAT, "o", color="black",
             label="Actual data", ms=0.7)

    # inspiration from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable

    class SafeEncoder(json.JSONEncoder):
        """JSON encoder for our data so that everything goes swellingly.
        Don't worry about this.
        """

        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return "%^" + str(obj.tolist()) + "%^"
            elif callable(obj):  # if is function
                return repr(obj)
            return json.JSONEncoder.default(self, obj)

    if args.outfile:
        with open(args.outfile, "w") as f:
            # recursively delete certain keys i.e. predictions

            def recursivelyRemove(obj, blacklist=[]):
                if isinstance(obj, dict):
                    for key in list(obj.keys()):
                        if key in blacklist:
                            del obj[key]
                        else:
                            obj[key] = recursivelyRemove(
                                obj[key], blacklist=blacklist)
                return obj

            outputData = recursivelyRemove(data, blacklist=[
                "predictions",
                "trig",
                "poly"
            ])

            outputJSON = json.dumps(outputData, cls=SafeEncoder,
                                    indent=4, sort_keys=True, ensure_ascii=False)

            outputJSON = outputJSON.replace("\"%^", "").replace("%^\"", "")
            f.write(outputJSON)

        print("\n"*2 + Fore.YELLOW + "  Equations and data written to `%s`." %
              args.outfile)

    if args.graph:
        plt.legend()
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "Runtime cancelled via KeyboardInterrupt.")
