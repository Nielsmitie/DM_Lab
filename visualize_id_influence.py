import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
import seaborn as sns
from argparse import ArgumentParser
import os

from helper import pandas_helper

path_to_result_file = 'results/id_estimation_influence.csv'
metrics = ['acc', 'r_square']
title = 'Visualization of the effects of the estimated intrinsic dimension: '
save_path = os.path.join('pics', 'id')

df = pandas_helper.pd_read_multi_column(path_to_result_file)
df[('config', 'id')] = df[('config', 'id')].str.replace('hardcode{\'id\': ', '').str.replace('}', '').astype('int')
df = df.groupby(('config', 'id')).mean()[['acc', 'r_square']]
df = df.T.reset_index(level=1, drop=True).T

if not os.path.exists(save_path):
    os.makedirs(save_path)

for metric in metrics:
    test = df[metric]

    test.T.sort_index().plot(title=title, logx=False, xlim=[0, 500])
    plt.ylabel(metric)
    plt.xlabel('Estimated ID')
    # plt.axvline(x=23, color='red')
    plt.scatter([23], test.loc[23], color='red')
    plt.axhline(y=test.loc[23], color='red')
    plt.savefig(os.path.join(save_path, title.strip().replace(':', '') + '_' + metric + '.png'), bbox_inches='tight')

    plt.show()