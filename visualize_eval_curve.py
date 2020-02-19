import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
import seaborn as sns
from argparse import ArgumentParser
import os

from helper import pandas_helper

path_to_result_file = 'results/curves.csv'


df = pandas_helper.pd_read_multi_column(path_to_result_file)

