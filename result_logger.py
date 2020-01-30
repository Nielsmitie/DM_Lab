import json
import pandas as pd
import os

from helper import pandas_helper


def save_result(config, results, dir_name=None, path_to_result_file='logs/results.csv'):
    # extract the parameters used for training
    training = pd.DataFrame.from_dict([config['training']])
    # extract the steps from the pipeline and merge them with the arguments used
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
    path_to_result_file = 'results/paper_agnos_s_2.csv'
    '''

    with open('configs/paper_config.json') as fr:
        config = json.load(fr)
    save_result(config, {'acc': {500: 0.5}, 'r_square': {500: 0.5}}, path_to_result_file=path_to_result_file)
    save_result(config, {'acc': {500: 0.5}, 'r_square': {500: 0.5}}, path_to_result_file=path_to_result_file)
    save_result(config, {'acc': {500: 0.5}, 'r_square': {500: 0.5}}, path_to_result_file=path_to_result_file)
    '''
    from helper.paper import datasets, acc_results, r2_results
    paper = pd.concat([pd.DataFrame(acc_results, index=datasets).T, pd.DataFrame(r2_results, index=datasets).T],
                      keys=['acc', 'r_square'], axis=1).stack()

    df = pandas_helper.pd_read_multi_column(path_to_result_file)
    mean = df[[('config', 'dataset'), ('acc', 100), ('r_square', 100)]].groupby(('config', 'dataset')).mean()

    mean = mean.T.reset_index(level=-1, drop=True).T
    mean.index = [i.replace("mat_loader{'name': '", "").replace("'}", "") for i in mean.index]

    result = paper.reset_index().join(mean, on='level_1', rsuffix='_experiment', lsuffix='_paper').set_index(['level_0', 'level_1'])

    result['deviance_acc'] = (1 - (result['acc_paper'] / result['acc_experiment'])) * 100
    result['deviance_r2'] = (1 - (result['r_square_paper'] / result['r_square_experiment'])) * 100

    results = result[['acc_paper', 'acc_experiment', 'r_square_paper', 'r_square_experiment']]
    deviance = result[['deviance_acc', 'deviance_r2']]
    print(results)
    print(deviance)
    std = df[[('config', 'dataset'), ('acc', 100), ('r_square', 100)]].groupby(('config', 'dataset')).std()
    std = std.T.reset_index(level=-1, drop=True).T
    print(std)
    # print(df)
