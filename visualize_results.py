import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
import seaborn as sns

from helper import pandas_helper

from helper.paper import datasets, acc_results, r2_results


def box_plot_results(df,
                     paper=None,
                     second_df=None,
                     metrics=None,
                     df_name='df',
                     second_df_name='other_df'):
    if metrics is None:
        metrics = ['acc', 'r_square']

    def format_df(tmp_df):
        counts = tmp_df['dataset'].value_counts()
        # build an index to pivot with. Build a new numeration for each dataset
        index = []
        for i, _ in enumerate(counts.index):
            index.extend(list(range(counts.values[i])))
        tmp_df['index'] = index
        return tmp_df

    def box_plot_df(tmp_df, hue, metric):
        plt.title(metric)
        test = list(tmp_df.columns.difference(['experiment']))
        if hue is None:
            ax = sns.boxplot(data=tmp_df, linewidth=1)
        else:
            ax = sns.boxplot(x='dataset', y='value', hue=hue, data=tmp_df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
        ax.tick_params(labelsize=9)

        return ax

    df = format_df(df)

    if second_df is not None:
        second_df = format_df(second_df)

    for metric in metrics:
        hue = None
        tmp = df.pivot(values=metric, columns='dataset', index='index').astype(float).assign(Experiment=df_name)

        if second_df is not None:
            tmp2 = second_df.pivot(values=metric, columns='dataset', index='index').astype(float).T.assign(Experiment=second_df_name).reset_index()
            tmp = df.pivot(values=metric, columns='dataset', index='index').astype(float).T.assign(Experiment=df_name).reset_index()
            cdf = pd.concat([tmp, tmp2])
            tmp = pd.melt(cdf, id_vars=['dataset', 'Experiment'], var_name=['Number'])
            hue = 'Experiment'

        ax = box_plot_df(tmp, hue=hue, metric=metric)

        # add the reference from the paper
        if paper is not None:
            test = paper[metric].loc[ax.get_xaxis().major.formatter.seq]
            plt.plot(test.values, '.', markersize=15, markerfacecolor='red')
        plt.show()

    print('test')


if __name__ == '__main__':
    model = 'spec'
    path_to_result_file = 'results/spec_train_test_reworked.csv'
    path_to_other_file = 'results/train_test_paper_agnos_s_reworked.csv'

    # load the hard coded paper results
    paper = pd.concat([pd.DataFrame(acc_results, index=datasets).T, pd.DataFrame(r2_results, index=datasets).T],
                      keys=['acc', 'r_square'], axis=1).stack()
    # select only the model that should be compared with the experimental results
    paper = paper.loc[model]

    # load and clean the experimental results
    def load_and_clean_dataset(path):
        tmp_df = pandas_helper.pd_read_multi_column(path)
        tmp_df = tmp_df[[('acc', 100), ('r_square', 100), ('config', 'dataset')]]
        tmp_df = tmp_df.T.reset_index(level=-1, drop=True).T
        tmp_df.columns = ['acc', 'r_square', 'dataset']
        tmp_df['dataset'] = tmp_df['dataset'].str.replace("mat_loader{'name': '", "").str.replace("'}", "")
        return tmp_df
    df = load_and_clean_dataset(path_to_result_file)
    second_df = load_and_clean_dataset(path_to_other_file)
    # plot the results in a boxplot
    box_plot_results(df, second_df=second_df, df_name='Spec train/test', second_df_name='Agnos s train/test')
    # box_plot_results(df, paper)
