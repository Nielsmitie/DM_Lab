import json
import pandas as pd
import os


def save_result(config, results, path_to_result_file='logs/results.csv'):
    # extract the parameters used for training
    training = pd.DataFrame.from_dict([config['training']])
    # extract the steps from the pipeline and merge them with the arguments used
    pipeline = pd.DataFrame({i: [str(config['pipeline'][i]) + str(config[i][config['pipeline'][i]])]
                             for i in ['dataset', 'normalize', 'id', 'model', 'score']})
    # combine both information in a one line pandas DataFrame
    one_liner = pd.concat([pipeline, training], axis=1)

    # merge the result from this run with results from the previous runs
    results = pd.DataFrame.from_dict(results)
    results = pd.concat([one_liner, results], axis=1)
    # if log dir does not exist
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    # if file does not exist create
    if not os.path.isfile(path_to_result_file):
        df = results
    else:
        df = pd.read_csv(path_to_result_file)
        df = df.append(results, ignore_index=True, sort=False)

    df.to_csv(path_to_result_file, index=False)


if __name__ == '__main__':
    # with open('configs/paper_config.json') as fr:
    #    config = json.load(fr)
    # save_result(config, {'acc': [str([500])], 'r_squared': [str([500])]})
    # save_result(config, {'acc': [str([500, 500])], 'r_squared': [str([500, 500])]})
    # save_result(config, {'acc': [str([500])], 'r_squared': [str([500])]})

    df = pd.read_csv('logs/results.csv')
    print(df)
