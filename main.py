import sys
import argparse
from tests import test
from modules.train import train
from data.processor import Processor
from modules.utils import open_configs

if __name__ == '__main__':
    cmd = sys.argv[1]
    parser = argparse.ArgumentParser()

    if cmd == 'process_data':
        data_config = open_configs(['data'])[0]
        processor = Processor(data_config)
        processor()

    elif cmd == 'train':
        parser.add_argument('-m', '--model', required=True)
        args, lf_args = parser.parse_known_args()
        train(args.model)

    elif cmd == 'test':
        parser.add_argument('-t', '--type', required=True)
        parser.add_argument('-c', '--ckpt', required=True)
        parser.add_argument('-i', '--image', required=True)
        args, lf_args = parser.parse_known_args()

        test(args.type, args.ckpt, args.image)