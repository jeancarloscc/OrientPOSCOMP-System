import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import services.function as sf
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import import_ipynb
import services.especialidades as esp
import datetime
import math
import services.function as fc
from textwrap import wrap
import seaborn as sns


warnings.filterwarnings('ignore')
