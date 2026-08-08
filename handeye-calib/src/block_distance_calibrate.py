#!/usr/bin/env python3
"""Fit monocular block distance models from --calib-record output."""

from __future__ import print_function

import argparse
import csv
import math

import numpy as np


def read_samples(paths, target):
    samples = []
    for path in paths:
        with open(path, "r", errors="ignore") as stream:
            for row in csv.reader(stream):
                if len(row) != 11 or row[1].strip() != target:
                    continue
                try:
                    known_z = float(row[0])
                    width = float(row[9])
                    height = float(row[10])
                except (TypeError, ValueError):
                    continue
                if all(math.isfinite(value) and value > 0.0
                       for value in (known_z, width, height)):
                    samples.append((known_z, width, height))
    if len(samples) < 6:
        raise RuntimeError("Need at least 6 valid samples for target %s." % target)
    if len(set(round(sample[0], 3) for sample in samples)) < 3:
        raise RuntimeError("Need samples from at least 3 different distances.")
    return samples
 
def aggregate_samples_by_distance(samples):
    groups = {}
    for known_z, width, height in samples:
        groups.setdefault(round(known_z, 3), []).append((known_z, width, height))
    aggregated = []
    for key in sorted(groups):
        group = np.asarray(groups[key], dtype=np.float64)
        aggregated.append(tuple(np.median(group, axis=0).tolist()))
    return aggregated


def fit_axis(samples, pixel_index):
    distances = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    pixels = np.asarray([sample[pixel_index] for sample in samples], dtype=np.float64)
    design = np.column_stack((1.0 / pixels, np.ones_like(pixels)))
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
        design, distances, rcond=None)
    predicted = design.dot(coefficients)
    errors = predicted - distances
    return {
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_error": float(np.max(np.abs(errors))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit width/height monocular distance calibration")
    parser.add_argument("--target", required=True,
                        choices=("power", "fire", "gas", "support"))
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    raw_samples = read_samples(args.files, args.target)
    samples = aggregate_samples_by_distance(raw_samples)
    width = fit_axis(samples, 1)
    height = fit_axis(samples, 2)
    print("%s:" % args.target)
    print("  width: {a: %.6f, b: %.6f}" % (width["a"], width["b"]))
    print("  height: {a: %.6f, b: %.6f}" % (height["a"], height["b"]))
    print("# frames=%d distance_points=%d width_rmse=%.2fmm width_max=%.2fmm "
          "height_rmse=%.2fmm height_max=%.2fmm" % (
              len(raw_samples), len(samples), width["rmse"], width["max_error"],
              height["rmse"], height["max_error"]))


if __name__ == "__main__":
    main()
