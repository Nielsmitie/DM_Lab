from argparse import ArgumentParser
import json
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import sklearn

import sys as sys

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, Callback

# Import all special directories
import dataset
import normalize
import id
import model
import regularizer
import score

"""
This document contains the execution of all necessary commands to run the AgnoS algorithm and its competitors.

It has an auto import function. Every file that is dropped into one of the directories is loaded and evaluated.
If it conforms to the specifications, then the module is added to the list of usable config options.

In a separate folder config files can be created. By command line options the config file is chosen.
The run is then performed by the specifications in the config file.

The syntax of the config file is as follows:
- pipeline defines for each step a file that is used to run the experiment.
- For each step there is an additional key where all available methods are listed. In this list the params for each
  method are defined. (It is possible to only list the used steps) (This is only a separate section to define params)
"""

# TODO: Use seeding/random states at every random operation and log them
# to make results replicable


def _register(module, func_name):
    """Register modul given a function name.

    Arguments:
        module {module} -- Module
        func_name {str} -- Name of the function used to call the module

    Returns:
        dict -- Registered functions
    """
    registered = {}
    for i in dir(module):
        if i[:1] == '_':
            continue
        if i in ['basename', 'dirname', 'isfile', 'join']:
            continue
        registered[i] = getattr(getattr(module, i), func_name)
    return registered


def parse_args():
    """Parse arguments from command line.

    Returns:
        Namespace -- Parsed arguments
    """
    args = ArgumentParser()
    args.add_argument('--debug', action='store_true')
    args.add_argument('--cpu', action='store_true', help='train on CPU')
    args.add_argument(
        '--config',
        type=str,
        default=os.path.join(
            'configs',
            'paper_config.json'))
    args.add_argument('--result_csv', type=str, default=None)
    return args.parse_args()


def main(args, config, result_csv='result.csv', log_level=logging.DEBUG):
    cfg_train = config['training']

    # for each python package import all functions with the name indicated in
    # the given string
    datasets = _register(dataset, 'get_dataset')
    normalizers = _register(normalize, 'normalize')
    id_estimators = _register(id, 'get_id')
    models = _register(model, 'get_model')
    regularizers = _register(regularizer, 'regularizer')
    scoring_functions = _register(score, 'score')
    competitors = ['SPEC', 'LAP', 'MCFS', 'NDFS', 'RANDOM']

    # specify log directory
    l = [config['pipeline']['model'], config['pipeline']['dataset'], datetime.now().strftime('%Y%m%d-%H%M%S')] + list(
        config['dataset'][config['pipeline']['dataset']].values())
    log_prefix = '_'.join(l)
    logdir = os.path.join('logs', log_prefix)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.basicConfig(
        filename=os.path.join(
            logdir,
            "logs.log"),
        level=log_level,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S')
    logging.info("Log directory: {}".format(logdir))

    """ Dataset loading """
    logging.info("Load dataset: {}".format(config['pipeline']['dataset']))
    X, y, num_classes = datasets[config['pipeline']['dataset']](
        **config['dataset'][config['pipeline']['dataset']])

    """ Normalize """
    X = normalizers[config['pipeline']['normalize']](
        X, **config['normalize'][config['pipeline']['normalize']])
    test_size = float(config['dataset']['test_split'])
    if test_size != 0. and test_size != 1.:
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y)
    else:
        X_train = X
        X_test = []
        y_train = y
        y_test = []

    """ ID estimation """
    estimated_id = id_estimators[config['pipeline']['id']](
        X_train, **config['id'][config['pipeline']['id']])
    logging.info("Estimated ID: {}".format(estimated_id))

    """ Competitors """
    if config['pipeline']['model'] in competitors:
        logging.info(
            'Competitor-Method: {}'.format(config['pipeline']['model']))
        logging.info("Calculating feature ranks.")
        sorted_features = models['competitors'](method=config['pipeline']['model'], X=X_train,
                                                n_clusters=num_classes,
                                                **config['model'][config['pipeline']['model']])
        """ Model evaluation """
        with open(os.path.join(logdir, "sorted_features.txt"), "w") as output:
            for feature in sorted_features:
                output.write(str(feature) + "\n")
        if len(X_test) == 0:
            evaluation(
                config,
                X_train,
                y_train,
                num_classes,
                sorted_features,
                logdir=logdir,
                result_csv=result_csv)
        else:
            evaluation(
                config,
                X_test,
                y_test,
                num_classes,
                sorted_features,
                logdir=logdir,
                result_csv=result_csv)
        return

    """ Autoencoder Model """
    """ Loss function and Compile """
    # extract the dictionary of the regularizers and feed them with the
    # parameter alpha (lambda in the paper)
    params = regularizers[config['pipeline']['regularizer']](
        **config['regularizer'][config['pipeline']['regularizer']])
    params.update(config['model'][config['pipeline']['model']])

    # tensorbord for logging
    tbc = TensorBoard(log_dir=logdir, write_images=True, update_freq='batch',
                      write_grads=True,
                      histogram_freq=config['training']['epochs'] // config['training']['hist_n_times'])

    # early stopping to reduce the number of epochs
    callbacks = [tbc]

    # custom callback to normalize the weights at the end of each epoch to one:
    class NormalizeWeights(Callback):
        # normalize the weights of the encoder layer at the end of an epoch to the length of 1
        def on_epoch_end(self, epoch, logs=None):
            # get the encoder layer. From there get the weights consisting of kernel and bias
            weights = self.model.get_layer('encoder').get_weights()
            weights[0] = weights[0] / np.sqrt(np.sum(np.square(weights[0]), keepdims=True))

            self.model.get_layer('encoder').set_weights(weights)

    if config['pipeline']['regularizer'] == 'reg_s' or \
            config['pipeline']['regularizer'] == 'reg_w' or \
            config['pipeline']['regularizer'] == 'reg_g':
        callbacks.append(NormalizeWeights())

    if cfg_train['patience'] != 0.:
        early_stopping = EarlyStopping(monitor='val_mean_squared_error', mode='min', restore_best_weights=True,
                                       patience=cfg_train['patience'])
        callbacks.append(early_stopping)

    # select the learner and hand over all parameters
    learner = models[config['pipeline']['model']](input_size=(X_train.shape[1],),
                                                  n_hidden=estimated_id,
                                                  metrics=cfg_train['metrics'],
                                                  lr=cfg_train['lr'],
                                                  **params)
    # save the layout and the number of parameters of the model to file
    with open(os.path.join(logdir, 'model_summary.txt'), 'w') as fw:
        learner.summary(print_fn=lambda x: fw.write(x + '\n'))

    learner.fit(X_train, X_train, batch_size=cfg_train['batch_size'], epochs=cfg_train['epochs'],
                callbacks=callbacks,
                validation_split=cfg_train['validation_split'], shuffle=True)

    """ Score Function """
    logging.info("Calculating feature ranks.")
    sorted_features = scoring_functions[config['pipeline']['score']](
        learner, **config['score'][config['pipeline']['score']])
    with open(os.path.join(logdir, "sorted_features.txt"), "w") as output:
        for feature in sorted_features:
            output.write(str(feature) + "\n")

    """ Model evaluation """
    if len(X_test) == 0:
        evaluation(
            config,
            X_train,
            y_train,
            num_classes,
            sorted_features,
            logdir=logdir,
            result_csv=result_csv)
    else:
        evaluation(
            config,
            X_test,
            y_test,
            num_classes,
            sorted_features,
            logdir=logdir,
            result_csv=result_csv)


def evaluation(config, X, y, num_classes, sorted_features, logdir, result_csv):
    """Evaluate the results achieved by feature selection using ACC and R².

    Arguments:
        config {dict} -- Config used for the run
        X {list} -- Dataset
        y {list} -- Labels
        num_classes {int} -- Number of classes
        sorted_features {list} -- List of features sorted starting with the feature having the highest impact
        logdir {str} -- Logging directory
        result_csv {str} -- Name of the result csv file
    """
    # write config file to log directory
    with open(os.path.join(logdir, 'config.json'), 'w') as fw:
        json.dump(config, fw, indent=4)

    # add all other evaluation functions here and log their results to
    # somewhere persistent
    from evaluation import k_means_accuracy, r_squared
    logging.info("Calculating ACC...")
    acc_scores = k_means_accuracy(X, y, num_clusters=num_classes, sorted_features=sorted_features,
                                  top_n=config['evaluation']['k_means_accuracy']['top_n'],
                                  repetitions=config['evaluation']['k_means_accuracy']['repetitions'])
    logging.info("ACC: {}".format(acc_scores))
    logging.info("Calculating R squared...")
    r_scores = r_squared(X, y, num_clusters=num_classes, sorted_features=sorted_features,
                         top_n=config['evaluation']['r_squared']['top_n'])
    logging.info("R squared: {}".format(r_scores))

    # global .csv that saves the results of all runs and makes them comparable
    from result_logger import save_result
    save_result(config, {'acc': acc_scores, 'r_square': r_scores}, dir_name=logdir,
                path_to_result_file=os.path.join('logs', result_csv))
    logging.info("Saved results to logs/{}".format(result_csv))


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Tensorflow and logging settings
    level = logging.INFO
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tensorflow debug messages
    if args.debug:
        level = logging.DEBUG
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        # suppress debug messages from pillow
        logging.getLogger('PIL').setLevel(logging.INFO)

    if args.cpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print('Train on CPU')

    if not tf.test.gpu_device_name():
        print('WARNING: No GPU available, use CPU instead...')

    # Load Config
    with open(args.config) as fr:
        config = json.load(fr)

    main(args, config, log_level=level, result_csv=args.result_csv)
