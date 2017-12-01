import numpy as np
import argparse
import tensorflow as tf
import sys

parser = argparse.ArgumentParser(description='argsparse test for tensorflow nightly build')
parser.add_argument('--myarg', type=str, help='fizz or buzz', default='fizz')
args = parser.parse_args()

def main(argv):
    print(args.myarg)

if __name__ == '__main__':
    tf.app.run()
