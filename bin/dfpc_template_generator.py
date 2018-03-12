#!/usr/bin/env python3
import os
import argparse
import yaml
from cdff_dev.code_generator import write_dfpc
from cdff_dev.path import check_cdffpath


def main():
    args = parse_args()
    check_cdffpath(args.cdffpath)
    with open(args.definition, "r") as f:
        node = yaml.load(f)
    write_dfn(node, args.output, args.source_folder, args.python_folder,
              args.cdffpath)


def parse_args():
    argparser = argparse.ArgumentParser(description="DFPC template generator")
    argparser.add_argument(
        "definition", type=str, help="DFPC definition")
    argparser.add_argument(
        "output", type=str, help="Target directory", nargs="?", default=".")
    argparser.add_argument(
        "--source_folder", type=str,
        help="Subdirectory of the output directory that will contain the "
             "source code template.", nargs="?", default=".")
    argparser.add_argument(
        "--python_folder", type=str,
        help="Subdirectory of the output directory that will contain the "
             "Python bindings.", nargs="?", default="python")
    argparser.add_argument(
        "--cdffpath", type=str, help="Path to CDFF repository.", nargs="?",
        default="CDFF")
    return argparser.parse_args()


if __name__ == "__main__":
    main()