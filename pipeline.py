from argparse import ArgumentParser
import json
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

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
    args.add_argument('--config', type=str, default=os.path.join('configs', 'test_config.json'))
    return args.parse_args()


def main(args, config):
    cfg_train = config['training']

    # for each python package import all functions with the name indicated in the given string
    datasets = _register(dataset, 'get_dataset')
    normalizers = _register(normalize, 'normalize')
    id_estimators = _register(id, 'get_id')
    models = _register(model, 'get_model')
    loss_functions = _register(loss, 'losses')
    scoring_functions = _register(score, 'score')

    """ Dataset loading """
    x, y, num_classes = datasets[config['pipeline']['dataset']](**config['dataset'][config['pipeline']['dataset']])

    """ Normalize """
    # TODO: split in train, test and fit_transform on train and transform on test.
    #  Is unsupervised, do we even need to split?
    x = normalizers[config['pipeline']['normalize']](x, **config['normalize'][config['pipeline']['normalize']])

    """ ID estimation """
    n_hidden = id_estimators[config['pipeline']['id']](x, **config['id'][config['pipeline']['id']])

    """ Auto-Encoder Model """
    
    """ Loss function and Compile """

    # specify log directory
    l = [config['pipeline']['model'], config['pipeline']['dataset'], datetime.now().strftime('%Y%m%d-%H%M%S')] + list(config['dataset'][config['pipeline']['dataset']].values())
    log_prefix = '_'.join(l)
    logdir = os.path.join('logs', log_prefix)
    logging.info(logdir)

    # write model summary to file
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # write config file to log directory
    with open(os.path.join(logdir, 'config.json'), 'w') as fw:
        json.dump(config, fw, indent=4)

    # this doesn't have to be done in the loop
    # extract the dictionary of the loss functions and feed them with the parameter alpha (lambda in the paper)
    params = loss_functions[config['pipeline']['loss']](**config['loss'][config['pipeline']['loss']])
    params.update(config['model'][config['pipeline']['model']])

    ranking_score = {}
    for i in range(cfg_train['repetitions']):
        # for each repetition a new file has to be created. Or in this case a new directory for the file is created
        repetition_dir = os.path.join('rep_' + str(i))
        # create the directory uf beeded
        if not os.path.exists(os.path.join(logdir, repetition_dir)):
            os.makedirs(os.path.join(logdir, repetition_dir))
        # tensorbord for logging
        # for each iteration a new logger has to be created for the new directory
        tbc = TensorBoard(log_dir=os.path.join(logdir, repetition_dir), write_images=True, update_freq='batch',
                          # record weight and gradient progress 10 times during training
                          write_grads=True,
                          histogram_freq=config['training']['epochs'] // config['training']['hist_n_times'])

        # early stopping to reduce the number of epochs
        # todo decide if restore_best_weights be True or False
        early_stopping = EarlyStopping(monitor='val_mean_squared_error', mode='min', restore_best_weights=False,
                                       patience=cfg_train['patience'])

        # select the learner and hand over all parameters
        learner = models[config['pipeline']['model']](input_size=(len(x[0]),),
                                                      n_hidden=n_hidden,
                                                      metrics=cfg_train['metrics'],
                                                      lr=cfg_train['lr'],
                                                      **params)
        # save the layout and the number of parameters of the model to file
        with open(os.path.join(logdir, 'model_summary.txt'), 'w') as fw:
            learner.summary(print_fn=lambda x: fw.write(x + '\n'))

        learner.fit(x, x, batch_size=cfg_train['batch_size'], epochs=cfg_train['epochs'],
                    callbacks=[early_stopping, tbc],
                    validation_split=cfg_train['validation_split'], shuffle=True)

        """ Score Function """
        # for each run the results are saved to ranking score and later averaged
        ranking_score['run_' + str(i)] = scoring_functions[config['pipeline']['score']](learner, **config['score'][config['pipeline']['score']])

    # convert to pandas to save the score as an .csv file
    df = pd.DataFrame.from_dict(ranking_score)

    # calculate average and standard deviation of the score
    df[['average', 'std']] = df.apply(lambda x: (np.mean(x), np.std(x)), result_type='expand', axis=1)
    # df = df.sort_values(by='features')

    # log the results to the log dir
    df.to_csv(os.path.join(logdir, 'run_results.csv'))
    print(df)

    """ Model evaluation """
    from evaluation import k_means_accuracy, r_squared
    # add all other evaluation functions here and log their results to somewhere persistent
    acc_scores = k_means_accuracy(x, y, num_clusters=num_classes, feature_rank_values=df['average'].values, top_n=config['evaluation']['k_means_accuracy']['top_n'])
    logging.info("ACC: {}".format(acc_scores))
    r_scores = r_squared(x, y, num_clusters=num_classes, feature_rank_values=df['average'].values, top_n=config['evaluation']['r_squared']['top_n'])
    logging.info("R²: {}".format(r_scores))


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
