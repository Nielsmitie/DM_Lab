import pandas as pd
import matplotlib.pyplot as plt


def extract_values(df, column, replace_before='{100: ', replace_after='}'):
    df[column] = df[column].str.replace(replace_before, '').str.replace(replace_after, '').astype('float')
    df = df[[column, 'dataset']].set_index('dataset').T
    return df


def extract_dataset(df, column='dataset', replace_before='mat_loader{\'name\': \'', replace_after='\'}'):
    df[column] = df[column].str.replace(replace_before, '').str.replace(replace_after, '')
    return df


def merge_own_and_paper(df):
    df = extract_dataset(df)
    df_acc = extract_values(df, 'acc')
    df_r_square = extract_values(df, 'r_square')
    # pretty print for comparison
    # acc table

    paper_results_acc = pd.DataFrame.from_dict({'arcene': [0.665],
                                                'Isolet': [0.526],
                                                'ORL': [0.570],
                                                'pixraw10P': [0.812],
                                                'Prostate-GE': [0.608],
                                                'TOX-171': [0.404],
                                                'warpPIE10P': [0.271],
                                                'Yale': [0.509]})
    df_acc = df_acc.append(paper_results_acc)
    df_acc.index = ['own', 'paper']
    df.name = 'ACC'

    paper_results_r_squared = pd.DataFrame.from_dict({'arcene': [0.6100, 0.460, 0.560, 0.490, 0.548],
                                                      'Isolet': [0.763, 0.762, 0.701, 0.747, 0.733],
                                                      'ORL': [0.800, 0.795, 0.780, 0.796, 0.769],
                                                      'pixraw10P': [0.855, 0.782, 0.832, 0.835, 0.761],
                                                      'Prostate-GE': [0.662, 0.620, 0.606, 0.614, 0.646],
                                                      'TOX-171': [0.581, 0.580, 0.528, 0.520, 0.559],
                                                      'warpPIE10P': [0.910, 0.897, 0.901, 0.904, 0.895],
                                                      'Yale': [0.703, 0.696, 0.671, 0.677, 0.659]})
    df_r_square = df_r_square.append(paper_results_r_squared)
    df_r_square.index = ['own', 'paper']
    df_r_square.name = 'R Squared'

    return pd.concat([df_acc, df_r_square], keys=['ACC', 'R Squared'])


def extract_knowledge(df):
    print(df)
    print('ACC')
    acc_pct_diff = df.loc['ACC'].pct_change().dropna() * 100
    print(acc_pct_diff)
    print('R Squared')
    r_square_pct_diff = df.loc['R Squared'].pct_change().dropna() * 100
    acc_pct_diff.T.plot(kind='bar')
    plt.show()
    print(r_square_pct_diff)


def plot_results():
    paper_results_r_squared = pd.DataFrame.from_dict({'arcene': [0.6100, 0.460, 0.560, 0.490, 0.548],
                                                      'Isolet': [0.763, 0.762, 0.701, 0.747, 0.733],
                                                      'ORL': [0.800, 0.795, 0.780, 0.796, 0.769],
                                                      'pixraw10P': [0.855, 0.782, 0.832, 0.835, 0.761],
                                                      'Prostate-GE': [0.662, 0.620, 0.606, 0.614, 0.646],
                                                      'TOX-171': [0.581, 0.580, 0.528, 0.520, 0.559],
                                                      'warpPIE10P': [0.910, 0.897, 0.901, 0.904, 0.895],
                                                      'Yale': [0.703, 0.696, 0.671, 0.677, 0.659]})
    paper_results_r_squared.index = ['Agnos-s', 'Agnos-w', 'Agnos-g', 'NDFS', 'SPEC']
    paper_results_r_squared = paper_results_r_squared / paper_results_r_squared.loc['NDFS'] - 1
    paper_results_r_squared.pct_change()
    paper_results_r_squared = paper_results_r_squared.loc[['Agnos-s', 'Agnos-w', 'Agnos-g', 'SPEC']]
    paper_results_r_squared.T.plot(kind='bar')
    plt.show()
