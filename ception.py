import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re

# what and where
inceptionV3_dir = 'graph'
images_dir = 'images'


dir_list = [x[0] for x in os.walk(images_dir)]
dir_list = dir_list[1:]
list_images = []
for image_sub_dir in dir_list:
	sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
	list_images.extend(sub_dir_images)

# extract features
#features, labels = extract_features(list_images)

from keras.applications.xception import Xception, preprocess_input, decode_predictions

pretrained = Xception(include_top = False, pooling='avg')
pretrained.summary()
