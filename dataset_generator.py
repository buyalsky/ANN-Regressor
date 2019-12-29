from pandas import DataFrame
import sys
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset generator")
    parser.add_argument("-f", "--file", required=True, help="dataset file to be generated")
    parser.add_argument("-l", "--length", type=int, default=50, help="length of dataset")
    args = parser.parse_args()

    x = [round(random.choice([-2, 2]) * random.random(), 1) for _ in range(args.length)]
    y = [round(random.choice([-2, 2]) * random.random(), 1) for _ in range(args.length)]

    z = [round(a ** 2 + b ** 2, 2) for a, b in zip(x, y)]
    values = {
        'x': x,
        'y': y,
        'z': z
    }
    FILENAME = args.file if sys.argv[1].endswith(".csv") else args.file + ".csv"

    df = DataFrame(values, columns=['x', 'y', 'z'])
    export_csv = df.to_csv(FILENAME, header=True)

    print(df)
