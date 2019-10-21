import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str)
parser.add_argument('--foo-abc', action='store_true')
print(sys.argv)
args = parser.parse_args()
print(args)
args_dict = vars(args)

print(args_dict)