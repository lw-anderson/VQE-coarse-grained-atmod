import argparse

from read_outputs import measure_all_outputs_concurrently

parser = argparse.ArgumentParser(description='Load outputs of optimisation routine and produce plots.')
parser.add_argument('--directory', type=str, help='Directory of output to read.')
parser.add_argument('--repeated_measurements', default=10, type=int,
                    help='Number of times to perform repeated measurement')

args = parser.parse_args()


def main():
    print('Performing subtraction measurements for outputs in')
    print(args.directory)
    results = measure_all_outputs_concurrently(args.directory, args.repeated_measurements)
    print(results)


if __name__ == '__main__':
    main()
