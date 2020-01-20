import pandas as pd


def pd_read_multi_column(path_to_file, *args, **kwargs):
    # read only the first two lines.
    df = pd.read_csv(path_to_file, header=[0, 1], skipinitialspace=True, *args, **kwargs)
    # Convert them to multiindex
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    acc = df['acc']
    acc.columns = acc.columns.astype('int')
    r = df['r_square']
    r.columns = r.columns.astype('int')

    df = pd.concat([acc, r, df['config']], axis=1, keys=['r_square', 'acc', 'config'])

    return df

