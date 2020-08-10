# -*- coding: utf-8 -*-
import argparse
import os
from benchmark import ImageBenchmark


def schedule_transfer_jobs(args):
    tag = args.tag
    bench = ImageBenchmark(datasets_dir='', models_dir='')
    for model_wrapper in bench.build_models():
        model_str = model_wrapper.__str__()
        model_str_tokens = model_str.split('-')
        if len(model_str_tokens) > 2 and model_str_tokens[-2].startswith('transfer'):
            print(f'{tag} benchmark.py \'-mask_str \"^{model_str}$\"\'')


def parse_args():
    """
    Parse command line input
    :return:
    """
    parser = argparse.ArgumentParser(description="Generate Philly job commands.")

    parser.add_argument("--tag", action="store", dest="tag", type=str, default="T",
                        help="A tag string to identify the job category.")
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    schedule_transfer_jobs(args)


if __name__ == "__main__":
    main()
