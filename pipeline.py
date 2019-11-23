from argparse import ArgumentParser
import json
import os
import logging
from datetime import datetime
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# in the following all special directories are imported
import dataset
import normalize
import id
import model
import loss
import score
"""
This document contains the execution of all necessary commands to run the agnos algorithm and its competitors.

It has an auto import function. Every file that is dropped into one of the directories is loaded and evaluated.
If it conforms to the specifications then the module is added to the list of usable config options.

In a separate folder config files can be created. By command line options the config file is chosen.
The run is then performed by the specifications in the config file.

The syntax of the config file is as follows:
1. pipeline defines for each step a file that is used to run the experiment.
2. For each step there is an additional key where all available methods are listed. In this list the params for each
    method are defined. (It is possible to only list the used steps) (This is only a separate section to define params)
3. training: define the hyperparameter used for training
"""


def _register(module, func_name):
    registered = {}
    for i in dir(module):
        if i[:1] == '_':
            continue
        if i in ['basename', 'dirname', 'isfile', 'join']:
            continue
        registered[i] = getattr(getattr(module, i), func_name)
    return registered


def parse_args():
    args = ArgumentParser()
    args.add_argument('--debug', action='store_true')
    args.add_argument('--cpu', action='store_true', help='train on CPU')
    args.add_argument('--config', type=str, default='configs/test_config.json')
    return args.parse_args()


def main(args, config):
    cfg_train = config['training']

    datasets = _register(dataset, 'get_dataset')
    normalizers = _register(normalize, 'normalize')
    id_estimators = _register(id, 'get_id')
    models = _register(model, 'get_model')
    loss_functions = _register(loss, 'losses')
    scoring_functions = _register(score, 'score')

    """ Dataset loading """
    x, y = datasets[config['pipeline']['dataset']](**config['dataset'][config['pipeline']['dataset']])

    """ Normalize """
    # TODO: split in train, test and fit_transform on train and transform on test
    x = normalizers[config['pipeline']['normalize']](x, **config['normalize'][config['pipeline']['normalize']])

    """ ID estimation """
    n_hidden = id_estimators[config['pipeline']['id']](x, **config['id'][config['pipeline']['id']])

    """ Auto-Encoder Model """
    
    """ Loss function and Compile """
    # TODO: fix input size parameter
    learner = models[config['pipeline']['model']](input_size=(4,),
                                                  n_hidden=n_hidden,
                                                  activation=config['model'][config['pipeline']['model']]['activation'],
                                                  loss=config['model'][config['pipeline']['model']]['loss'],
                                                  metrics=cfg_train['metrics'],
                                                  lr=cfg_train['lr'],
                                                  **loss_functions[config['pipeline']['loss']]())
    # specify log directory
    l = [config['pipeline']['model'], config['pipeline']['dataset'], datetime.now().strftime('%Y%m%d-%H%M%S')] + list(config['dataset'][config['pipeline']['dataset']].values())
    log_prefix = '_'.join(l)
    logdir = os.path.join('logs', log_prefix)
    logging.info(logdir)

    # write model summary to file
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, 'model_summary.txt'), 'w') as fw:
        learner.summary(print_fn=lambda x: fw.write(x + '\n'))

    # write config file to log directory
    with open(os.path.join(logdir, 'config.json'), 'w') as fw:
        json.dump(config, fw, indent=4)

    # tensorbord for logging
    tbc = TensorBoard(log_dir=logdir, write_images=True, update_freq='batch')
    file_writer = tf.summary.create_file_writer(logdir + '/metrics')
    file_writer.set_as_default()

    learner.fit(x, x, batch_size=cfg_train['batch_size'], epochs=cfg_train['epochs'], callbacks=[tbc],
                validation_split=cfg_train['validation_split'], shuffle=True)

    """ Score Function """
    ranking_score = scoring_functions[config['pipeline']['score']](learner, **config['score'][config['pipeline']['score']])

    df = pd.DataFrame(ranking_score, columns=['features'])
    df = df.sort_values(by='features')
    print(df)

    """ Model evaluation """
    # TODO: implement methods for final model evaluation


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

    main(args, config)
