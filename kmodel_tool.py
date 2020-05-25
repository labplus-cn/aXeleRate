import argparse
import os
import sys
from axelerate import setup_training, setup_inference
import json
from axelerate.networks.common_utils.convert import Converter

__version__ = "1.0"
__prog__ = sys.argv[0]
# Function


def train_model(args):
    config = json.load(args.config)
    model_path = setup_training(config_dict=config)
    # Verify model accuracy
    if args.infer:
        setup_inference(config, model_path)

def convert_k210(args):
    convert_model = Converter("k210")
    convert_model.convert_k210(args.model_path, args.dataset)

def version(args):
    print(__version__)


# argparse

def main():
    parser = argparse.ArgumentParser(description='Tools for trained model,conversion to kmodel(K210).')
    subparsers = parser.add_subparsers(dest='sub_program', help='Run %s {command} -h for additional help' % (parser.prog))

    cmd_train = subparsers.add_parser('train', help='Training kmodel')
    # cmd_train.add_argument("-t", "--type", choices=["classifier", "yolo"], required=True, help="Model architecture")
    cmd_train.add_argument("-c", "--config", type=argparse.FileType('r'), required=True, help="Config json file path")
    cmd_train.add_argument("-i", "--infer", type=bool, default=False, help="Infer model accuracy")
    cmd_train.set_defaults(func=train_model)

    convert_cmd = subparsers.add_parser("convert", help="Convert tflite to kmdodel")
    convert_cmd.add_argument("model_path", help="tflite model path")
    convert_cmd.add_argument("dataset", help="dataset path")
    convert_cmd.set_defaults(func=convert_k210)

    version_cmd = subparsers.add_parser("version", help="Print version")
    version_cmd.set_defaults(func=version)

    args = parser.parse_args()

    if args.sub_program is None:
        parser.print_help()
        parser.exit(1)

    args.func(args)
    print(args)

def _main():
    try:
        main()
    except Exception as e:
        print('\n', e)
        sys.exit(2)


if __name__ == '__main__':
    _main()
