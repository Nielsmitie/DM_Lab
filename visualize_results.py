import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle

from helper import pandas_helper

from helper.paper import datasets, acc_results, r2_results

plt.style.use('fivethirtyeight')
path_to_result_file = 'logs/paper_agnos_s_reworked.csv'
model = 'agnos_s'


paper = pd.concat([pd.DataFrame(acc_results, index=datasets).T, pd.DataFrame(r2_results, index=datasets).T],
                  keys=['acc', 'r_square'], axis=1).stack()
paper = paper.loc['agnos_s']

df = pandas_helper.pd_read_multi_column(path_to_result_file)
df = df[[('acc', 100), ('r_square', 100), ('config', 'dataset')]]
df = df.T.reset_index(level=-1, drop=True).T
df.columns = ['acc', 'r_square', 'dataset']

df['dataset'] = df['dataset'].str.replace("mat_loader{'name': '", "").str.replace("'}", "")


test = [i for i in range(10)]
index = []
for i in range(8):
    index.extend(test)

df['index'] = index

acc = df.pivot(values='acc', columns='dataset', index='index').astype(float)
r2 = df.pivot(values='r_square', columns='dataset', index='index').astype(float)
test = list(acc.columns)

ax = acc.boxplot(fontsize=9)
plt.show()

r2.boxplot()
plt.show()

print('test')
