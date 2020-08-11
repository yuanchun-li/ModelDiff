# -*- coding: utf-8 -*-
# sample usage:
# python .\ModelDiff_code\schedule_jobs.py -phase train -prefix 'python philly_ycli/submit_job.py -params'
import argparse
import os
import re
from benchmark import ImageBenchmark


def schedule_one_on_one(args):
    bench = ImageBenchmark(datasets_dir='', models_dir='')
    for model_wrapper in bench.list_models():
        model_str_tokens = model_wrapper.__str__().split('-')
        if len(model_str_tokens) >= 2 and model_str_tokens[-2].startswith(args.phase):
            model_str = re.sub(r'[^A-Za-z0-9.]+', '_', model_wrapper.__str__())
            print(f'{args.prefix} \'-regenerate -mask _{model_str}_\'')


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
    parser.add_argument("-mode", action="store", dest="mode", type=str, default="1@1",
                        help="Schedule mode. 1@1 means one job on one device. all@1 means all on one.")
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    if args.mode == "1@1":
        schedule_one_on_one(args)
    elif args.mode == "all@1":
        schedule_all_on_one(args)


if __name__ == "__main__":
    main()

