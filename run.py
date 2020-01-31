import json
import os
import logging
import tensorflow as tf
from argparse import ArgumentParser

from pipeline import main
from helper import paper


def run_experiments(config, experiment_name, repeats):
    """Run the experiments with the given parameters and hardcoded id.
    
    Arguments:
        config {dict} -- Loaded config file
        experiment_name {str} -- Name of the results file
        repeats {int} -- Number of repeats
    """    
    for dataset, id in paper.ids.items():
        for repeat in range(repeats):
            config['dataset'][config['pipeline']['dataset']]['name'] = dataset
            config['id']['hardcode']['id'] = id
            print('{}/{} running with {} and id {}'.format(repeat+1, repeats, dataset, id))
            main([], config, experiment_name)


def parse_args():
    """Parse arguments from command line.
    
    Returns:
        Namespace -- Parsed arguments
    """    
    args = ArgumentParser()
    args.add_argument('--config', type=str, default=os.path.join('configs', 'paper_config.json'))
    args.add_argument('--repeats', type=int, default=10)
    args.add_argument('--debug', action='store_true')
    args.add_argument('--cpu', action='store_true', help='train on CPU')
    args.add_argument('--experiment_name', type=str, default='results.csv')

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()

    level = logging.INFO
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorflow debug messages
    if args.debug:
        level = logging.DEBUG
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        logging.getLogger('PIL').setLevel(logging.INFO)  # suppress debug messages from pillow
    logging.basicConfig(level=level, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logging.info('train on CPU')

    if not tf.test.gpu_device_name():
        logging.warning('no GPU available, use CPU instead...')

    with open(args.config) as fr:
        config = json.load(fr)

    run_experiments(config, args.experiment_name, args.repeats)
