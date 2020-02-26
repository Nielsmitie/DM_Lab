import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
import seaborn as sns
from argparse import ArgumentParser
import os

from helper import pandas_helper

path_to_result_file = 'results/curves.csv'
metrics = ['acc', 'r_square']
title = 'Results with different top n features for evaluation: '
save_path = os.path.join('pics', 'curve')

df = pandas_helper.pd_read_multi_column(path_to_result_file)
df[('config', 'model')] = df[('config', 'model')].str.replace('{.*}', '')
df.set_index(('config', 'model'), inplace=True)

if not os.path.exists(save_path):
    os.makedirs(save_path)

for metric in metrics:
    test = df[metric]

    test.T.sort_index().plot(title=title)
    plt.ylabel(metric)
    plt.xlabel('Top n features')

    plt.savefig(os.path.join(save_path, title.strip().replace(':', '') + '_' + metric + '.png'), bbox_inches='tight')

    plt.show()
