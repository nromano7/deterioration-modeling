import sys
print("Python version: {}".format(sys.version))

import IPython
print("IPython version: {}".format(IPython.__version__))

import pandas as pd
print("pandas version: {}".format(pd.__version__) + " name: pd")

import numpy as np
print("NumPy version: {}".format(np.__version__) + " name: np")

import scipy as sp
print("SciPy version: {}".format(sp.__version__) + " name: sp")

import sklearn
print("scikit-learn version: {}".format(sklearn.__version__)+ " name: skl")

import plotly
import plotly.offline as plt
import plotly.graph_objs as go

print("plotly offline version: {}".format(plotly.__version__) + " name: plt")
print("\tplotly graph object imported as: go")
print("\tNotebook Mode Initiated")

import matplotlib as mpl
print("matplotlib version: {}".format(mpl.__version__) + " name: mpl")

import pickle
print("pickle imported as: pickle")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

plt.init_notebook_mode(connected=True)