import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score, StratifiedShuffleSplit
)
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier
import services.function as sf
import seaborn.objects as so
import datetime
import math
from textwrap import wrap
import pickle

warnings.filterwarnings('ignore')
