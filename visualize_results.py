import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
import seaborn as sns
from argparse import ArgumentParser
import os

from helper import pandas_helper

from helper.paper import datasets, acc_results, r2_results


def box_plot_results(df,
                     paper=None,
                     second_df=None,
                     metrics=None,
                     df_name='df',
                     second_df_name='other_df',
                     title='Title template: ',
                     save_path='pics'):
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
        plt.title(title + metric)
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

        if second_df is not None:
            # prepare both datasets and melt them together
            tmp2 = second_df.pivot(values=metric, columns='dataset', index='index').astype(float).T.assign(Experiment=second_df_name).reset_index()
            tmp = df.pivot(values=metric, columns='dataset', index='index').astype(float).T.assign(Experiment=df_name).reset_index()
            cdf = pd.concat([tmp, tmp2])
            tmp = pd.melt(cdf, id_vars=['dataset', 'Experiment'], var_name=['Number'])
            hue = 'Experiment'
        else:
            tmp = df.pivot(values=metric, columns='dataset', index='index').astype(float)

        ax = box_plot_df(tmp, hue=hue, metric=metric)

        # add the reference from the paper
        if paper is not None:
            test = paper[metric].loc[ax.get_xaxis().major.formatter.seq]
            plt.plot(test.values, '.', markersize=15, markerfacecolor='red')

        if save_path is None:
            plt.show()
        else:
            path = os.path.join(save_path, df_name)
            file_name = title + '_' + metric + '.png'
            print('saving plot to: {}'.format(os.path.join(path, file_name)))
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, file_name))
            plt.show()

    print('test')


def parse_args():
    # Parse arguments
    args = ArgumentParser()
    args.add_argument(
        '--path',
        type=str,
        default=os.path.join(
            'results',
            'results.csv'))
    args.add_argument(
        '--model',
        type=str,
        default=None
    )
    args.add_argument(
        '--second_path',
        type=str,
        default=None
    )
    args.add_argument(
        '--title',
        type=str,
        default='Title template: '
    )
    args.add_argument(
        '--df_name',
        type=str,
        default='df'
    )
    args.add_argument(
        '--other_df_name',
        type=str,
        default='other_df'
    )
    args.add_argument(
        '--save',
        type=str,
        default=None
    )
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()

    path_to_result_file = args.path

    model = args.model
    path_to_other_file = args.second_path

    df_name = args.df_name
    second_df_name = args.other_df_name

    title = args.title
    if title is None:
        title = path_to_result_file

    save_path = args.save

    if model:
        # load the hard coded paper results
        paper = pd.concat([pd.DataFrame(acc_results, index=datasets).T, pd.DataFrame(r2_results, index=datasets).T],
                          keys=['acc', 'r_square'], axis=1).stack()
        # select only the model that should be compared with the experimental results
        paper = paper.loc[model]
    else:
        paper = None

    # load and clean the experimental results
    def load_and_clean_dataset(path):
        tmp_df = pandas_helper.pd_read_multi_column(path)
        tmp_df = tmp_df[[('acc', 100), ('r_square', 100), ('config', 'dataset')]]
        tmp_df = tmp_df.T.reset_index(level=-1, drop=True).T
        tmp_df.columns = ['acc', 'r_square', 'dataset']
        tmp_df['dataset'] = tmp_df['dataset'].str.replace("mat_loader{'name': '", "").str.replace("'}", "")
        return tmp_df
    df = load_and_clean_dataset(path_to_result_file)
    if path_to_other_file:
        second_df = load_and_clean_dataset(path_to_other_file)
    else:
        second_df = None
    # plot the results in a boxplot
    box_plot_results(df, second_df=second_df,
                     df_name=df_name,
                     second_df_name=second_df_name,
                     paper=paper,
                     title=title,
                     save_path=save_path)
