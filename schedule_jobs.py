# -*- coding: utf-8 -*-
import argparse
import os
import re
from benchmark import ImageBenchmark


def schedule_one_on_one(args):
    bench = ImageBenchmark(datasets_dir='', models_dir='')
    for model_wrapper in bench.build_models():
        model_str_tokens = model_wrapper.__str__().split('-')
        if len(model_str_tokens) >= 2 and model_str_tokens[-2].startswith(args.phase):
            model_str = re.sub(r'[^A-Za-z0-9.]+', '_', model_wrapper.__str__())
            print(f'{args.prefix} \'-mask _{model_str}_\'')


def schedule_all_on_one(args):
    print(f'{args.prefix} \'-phase {args.phase}\'')


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Generate Philly job commands.")

    parser.add_argument("-prefix", action="store", dest="prefix", type=str, default="T",
                        help="A prefix string for the command.")
    parser.add_argument("-phase", action="store", dest="phase", type=str, default="",
                        help="The phase to run. Use a prefix to filter the phases.")
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    schedule_one_on_one(args)


if __name__ == "__main__":
    main()
