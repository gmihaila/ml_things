# coding=utf-8
# Copyright 2020 George Mihaila.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions related to plotting"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, classes='', normalize=False, title=None, cmap=plt.cm.Blues, image=None,
                          verbose=0, magnify=1.2, dpi=50):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    y_true needs to contain all possible labels.

    :param y_true: array labels values.
    :param y_pred: array predicted label values.
    :param classes: array list of label names.
    :param normalize: bool normalize confusion matrix or not.
    :param title: str string title of plot.
    :param cmap: plt.cm plot theme
    :param image: str path to save plot in an image.
    :param verbose: int print confusion matrix when calling function.
    :param magnify: int zoom of plot.
    :param dpi: int clarity of plot.
    :return: array confusion matrix used to plot.

    Note:
        - Plot themes:
        cmap=plt.cm.Blues - used as default.
        cmap=plt.cm.BuPu
        cmap=plt.cm.GnBu
        cmap=plt.cm.Greens
        cmap=plt.cm.OrRd
    """
    if len(y_true) != len(y_pred):
        # make sure lengths match
        raise ValueError("`y_true` needs to have same length as `y_pred`!")

    # Class labels setup. If none, generate from y_true y_pred
    classes = list(classes)
    if classes:
        assert len(set(y_true)) == len(classes)
    else:
        classes = set(y_true)
    # Title setup
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Normalize setup
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # Print if verbose
    if verbose > 0:
        print(cm)
    # Plot setup
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.grid(False)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig = plt.gcf()
    figsize = fig.get_size_inches()
    fig.set_size_inches(figsize * magnify)
    if image:
        fig.savefig(image, dpi=dpi)
    return cm
