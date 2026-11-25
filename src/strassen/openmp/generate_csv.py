"""Generate two square input matrices for the OpenMP runner."""

import argparse
import csv
import random


def main():
    parser = argparse.ArgumentParser(description="Write A.csv and B.csv in the current directory.")
    parser.add_argument("size", type=int, help="number of rows and columns in each matrix")
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size must be positive")

    rng = random.Random(1)
    for filename in ("A.csv", "B.csv"):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            for _ in range(args.size):
                writer.writerow(f"{rng.random():.8f}" for _ in range(args.size))


if __name__ == "__main__":
    main()
