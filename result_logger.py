import json
import pandas as pd
import os
from argparse import ArgumentParser

from helper import pandas_helper


def save_result(config, results, dir_name=None,
                path_to_result_file='logs/results.csv'):
    """Save results to central csv.

    Arguments:
        config {dict} -- Dict containing the config
        results {dict} -- Dict containing acc and r_square and their values

    Keyword Arguments:
        dir_name {str} -- Name of the directory to save in (default: {None})
        path_to_result_file {str} -- Path to result file (default: {'logs/results.csv'})
    """
    # extract the parameters used for training
    training = pd.DataFrame.from_dict([config['training']])
    # extract the steps from the pipeline and merge them with the arguments
    # used
    pipeline = pd.DataFrame({i: [str(config['pipeline'][i]) + str(config[i][config['pipeline'][i]])]
                             for i in ['dataset', 'normalize', 'id', 'model', 'score']})
    # combine both information in a one line pandas DataFrame
    one_liner = pd.concat([pipeline, training], axis=1)

    # merge the result from this run with results from the previous runs
    # for each metric create a dataframe
    metrics = {}
    tmp = pd.DataFrame.from_dict(results)
    for i in results.keys():
        metrics[i] = tmp[[i]].T.reset_index(drop=True)
    # add the config
    metrics['config'] = one_liner
    # merge them with a multiindex
    results = pd.concat(metrics.values(), keys=metrics.keys(), axis=1)
    if dir_name:
        results.index = [dir_name]
    # results.columns = pd.MultiIndex.from_tuples(results.columns)
    # now all config is found under the multiindex column config, and all accuracy is stored by the
    # number of top k features

    # if log dir does not exist
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    # if file does not exist create
    if not os.path.isfile(path_to_result_file):
        df = results
    else:
        # now it is the challenge to retrieve the multicolumn dataframe
        df = pandas_helper.pd_read_multi_column(path_to_result_file)
        # df.T[1] = results.T
        # results.reset_index().join(df, on=df.index).set_index(results.index.names)
        df = results.T.reset_index().join(df.T, on=['level_0', 'level_1'], lsuffix='ka', how='outer')\
            .set_index(['level_0', 'level_1']).T
        # df = pd.concat([df, results], ignore_index=True, sort=False)

    df.to_csv(path_to_result_file, index=False)


if __name__ == '__main__':
    # Parse arguments
    args = ArgumentParser()
    args.add_argument(
        '--path',
        type=str,
        default=os.path.join(
            'results',
            'results.csv'))
    args = args.parse_args()
    path_to_result_file = args.path

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from helper.paper import datasets, acc_results, r2_results
    paper = pd.concat([pd.DataFrame(acc_results, index=datasets).T, pd.DataFrame(r2_results, index=datasets).T],
                      keys=['acc', 'r_square'], axis=1).stack()

    df = pandas_helper.pd_read_multi_column(path_to_result_file)
    mean = df[[('config', 'dataset'), ('acc', 100), ('r_square', 100)]].groupby(
        ('config', 'dataset')).mean()

    mean = mean.T.reset_index(level=-1, drop=True).T
    mean.index = [
        i.replace(
            "mat_loader{'name': '",
            "").replace(
            "'}",
            "") for i in mean.index]

    result = paper.reset_index().join(mean, on='level_1', rsuffix='_experiment', lsuffix='_paper').set_index(['level_0', 'level_1'])

    result['deviance_acc'] = (1 - (result['acc_paper'] / result['acc_experiment'])) * 100
    result['diff_acc'] = result['acc_paper'] - result['acc_experiment']
    result['deviance_r2'] = (1 - (result['r_square_paper'] / result['r_square_experiment'])) * 100
    result['diff_r2'] = result['r_square_paper'] - result['r_square_experiment']

    results = result[['acc_paper', 'acc_experiment',
                      'r_square_paper', 'r_square_experiment']]
    deviance = result[['deviance_acc', 'deviance_r2']]
    diff = result[['diff_acc', 'diff_r2']]
    print(results)
    print(deviance)
    print(diff)
    std = df[[('config', 'dataset'), ('acc', 100), ('r_square', 100)]
             ].groupby(('config', 'dataset')).std()
    std = std.T.reset_index(level=-1, drop=True).T
    print(std)
