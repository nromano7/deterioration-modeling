import sys
print("Python version: {}".format(sys.version))

import pandas as pd
print("pandas version: {}".format(pd.__version__) + " name: pd")

import numpy as np
print("NumPy version: {}".format(np.__version__) + " name: np")

import scipy as sp
print("SciPy version: {}".format(sp.__version__) + " name: sp")

import IPython
print("IPython version: {}".format(IPython.__version__))

import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))

import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))

import plotly 
print("plotly version: {}".format(plotly.__version__))
