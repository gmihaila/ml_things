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
import warnings
from sklearn.metrics import confusion_matrix


def plot_array(array, start_step=0, step_size=1, use_label=None, use_title=None, points_values=False, points_round=3,
               use_xlabel=None, use_xticks=True, use_ylabel=None, style_sheet='ggplot', use_grid=True,
               use_linestyle='-', width=3, height=1, magnify=1.2, use_dpi=50, path=None, show_plot=True):
    r"""
    Create plot from a single array of values.

    Arguments:

        array (:obj:`list / np.ndarray`):
            List of values. Can be of type list or np.ndarray.

        start_step (:obj:`int`, `optional`, defaults to :obj:`0`):
            Value to offset all x-axis values. This argument is optional and it has a default value attributed inside
            the function.

        step_size (:obj:`int`, `optional`, defaults to :obj:`1`):
            This argument is optional and it has a default value attributed inside the function. It will multiply
            each x-axis position value to it.

        use_label (:obj:`str`, `optional`):
            Display label on plot for whole array. This argument is optional and it will have a `None` value attributed
            inside the function.

        use_title (:obj:`str`, `optional`):
            Title on top of plot. This argument is optional and it will have a `None` value attributed inside the
            function for which it won't display any title.

        points_values (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Display each point value on the plot. This argument is optional and it has a default value attributed
            inside the function.

        points_round (:obj:`int`, `optional`, defaults to :obj:`1`):
            Round decimal valus for points values. This argument is optional and it has a default value attributed
            inside the function.

        use_xlabel (:obj:`str`, `optional`):
            Label to use for x-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.

        use_xticks (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Display x-axis tick values. This argument is optional and it has a default value attributed
            inside the function.

        use_ylabel (:obj:`str`, `optional`):
            Label to use for y-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.

        style_sheet (:obj:`str`, `optional`, defaults to :obj:`ggplot`):
            Style of plot. Use plt.style.available to show all styles. This argument is optional and it has a default
            value attributed inside the function.

        use_grid (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Show grid on plot or not. This argument is optional and it has a default value attributed inside
            the function.

        use_linestyle (:obj:`str`, `optional`, defaults to :obj:`-`):
            Style to use on line from ['-', '--', '-.', ':']. This argument is optional and it has a default
            value attributed inside the function.

        width (:obj:`int`, `optional`, defaults to :obj:`3`):
            Horizontal length of plot. This argument is optional and it has a default value attributed inside
            the function.

        height (:obj:`int`, `optional`, defaults to :obj:`1`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        magnify (:obj:`int`, `optional`, defaults to :obj:`1.2`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        use_dpi (:obj:`int`, `optional`, defaults to :obj:`50`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        path (:obj:`str`, `optional`):
            Vertical length of plot. This argument is optional and it will have a None value attributed inside
            the function.

        show_plot (:obj:`bool`, `optional`, defaults to :obj:`True`):
            if you want to call `plt.show()`. or not (if you run on a headless server). This argument is optional and
            it has a default value attributed inside the function.

    Raises:

        ValueError: If `array` is not of type `list` or `np.ndarray`

        ValueError: If `style_sheet` is not valid.

        ValueError: If `use_linestyle` is not valid.

    """

    # check if `array` is correct format
    if not isinstance(array, list) or isinstance(array, np.ndarray):
        # raise value error
        raise ValueError("`array` needs to be a list of values!")

    if style_sheet in plt.style.available:
        # set style of plot
        plt.style.use(style_sheet)
    else:
        # style is not correct
        raise ValueError("`style_sheet=%s` is not in the supported styles: %s" % (str(style_sheet),
                                                                                  str(plt.style.available)))
    # all linestyles
    linestyles = ['-', '--', '-.', ':']

    # check if linestyle is set right
    if use_linestyle not in linestyles:
        # raise error
        raise ValueError("`linestyle=%s` is not in the styles: %s!" % (str(use_linestyle), str(linestyles)))

    # set steps plotted on x-axis - we can use step if 1 unit has different value

    if start_step > 0:
        # Offset all steps by start_step.
        steps = np.array(range(0, len(array))) * step_size + start_step
    else:
        steps = np.array(range(1, len(array) + 1)) * step_size
    # single plot figure
    plt.subplot(1, 2, 1)
    # plot array as a single line
    plt.plot(steps, array, linestyle=use_linestyle, label=use_label)
    # Plots points values
    if points_values:
        # Loop through each point and plot the label.
        for x, y in zip(steps, array):
            # Add text label to plo.
            plt.text(x, y, str(round(y, points_round)))
    # set title of figure
    plt.title(use_title)
    # set horizontal axis name
    plt.xlabel(use_xlabel)
    # Use x ticks with steps.
    plt.xticks(steps) if use_xticks else None
    # set vertical axis name
    plt.ylabel(use_ylabel)
    # place legend best position
    plt.legend(loc='best') if use_label is not None else None
    # display grid depending on `use_grid`
    plt.grid(use_grid)
    # make figure nice
    plt.tight_layout()
    # get figure object from plot
    fig = plt.gcf()
    # get size of figure
    figsize = fig.get_size_inches()
    # change size depending on height and width variables
    figsize = [figsize[0] * width * magnify, figsize[1] * height * magnify]
    # set the new figure size
    fig.set_size_inches(figsize)
    # save figure to image if path is set
    fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None
    # show plot
    plt.show() if show_plot is True else None

    return


def plot_dict(dict_arrays, start_step=0, step_size=1, use_title=None, points_values=False, points_round=3,
              use_xlabel=None, use_ylabel=None,
              style_sheet='ggplot', use_grid=True, use_linestyles=None, width=3, height=1, magnify=1.2,
              use_dpi=50, path=None, show_plot=True):
    r"""
    Create plot from a single array of values.

    Arguments:

        dict_arrays (:obj:`dict([list])`):
            Dictionary of arrays that will get plotted. The keys in dictionary are used as labels and the values as
            arrays that get plotted.

        start_step (:obj:`int`, `optional`, defaults to :obj:`0`):
            Starting value of plot.This argument is optional and it has a default value attributed inside
            the function.

        step_size (:obj:`int`, `optional`, defaults to :obj:`q`):
            Steps shows on x-axis. Change if each steps is different than 1.This argument is optional and it has a
            default value attributed inside the function.

        use_title (:obj:`int`, `optional`):
            Title on top of plot. This argument is optional and it will have a `None` value attributed
            inside the function.

        points_values (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Display each point value on the plot. This argument is optional and it has a default value attributed
            inside the function.

        points_round (:obj:`int`, `optional`, defaults to :obj:`1`):
            Round decimal valus for points values. This argument is optional and it has a default value attributed
            inside the function.

        use_xlabel (:obj:`str`, `optional`):
            Label to use for x-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.

        use_ylabel (:obj:`str`, `optional`):
            Label to use for y-axis value meaning. This argument is optional and it will have a `None` value attributed
            inside the function.

        style_sheet (:obj:`str`, `optional`, defaults to :obj:`ggplot`):
            Style of plot. Use plt.style.available to show all styles. This argument is optional and it has a default
            value attributed inside the function.

        use_grid (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Show grid on plot or not. This argument is optional and it has a default value attributed inside
            the function.

        use_linestyles (:obj:`str`, `optional`, defaults to :obj:`-`):
            Style to use on line from ['-', '--', '-.', ':']. This argument is optional and it has a default
            value attributed inside the function.

        width (:obj:`int`, `optional`, defaults to :obj:`3`):
            Horizontal length of plot. This argument is optional and it has a default value attributed inside
            the function.

        height (:obj:`int`, `optional`, defaults to :obj:`1`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        magnify (:obj:`int`, `optional`, defaults to :obj:`1.2`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        use_dpi (:obj:`int`, `optional`, defaults to :obj:`50`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        path (:obj:`str`, `optional`):
            Vertical length of plot. This argument is optional and it will have a None value attributed inside
            the function.

        show_plot (:obj:`bool`, `optional`, defaults to :obj:`True`):
            if you want to call `plt.show()`. or not (if you run on a headless server). This argument is optional and
            it has a default value attributed inside the function.

    Raises:

        ValueError: If `dict_arrays` is not of type `dictionary`.

        ValueError: If `dict_arrays` doesn't have string keys.

        ValueError: If `dict_arrays` doesn't have array values.

        ValueError: If `style_sheet` is not valid.

        ValueError: If `use_linestyle` is not valid.

        ValueError: If `points_values`of type list don't have same length as `dict_arrays`.

    """
    # check if `dict_arrays` is correct format
    if not isinstance(dict_arrays, dict):
        # raise value error
        raise ValueError("`dict_arrays` needs to be a dictionary of values!")
    for label, array in dict_arrays.items():
        # check if format is correct
        if not isinstance(label, str):
            # raise value error
            raise ValueError("`dict_arrays` needs string keys!")
        if not isinstance(array, list) or isinstance(array, np.ndarray):
            # raise value error
            raise ValueError("`dict_arrays` needs lists values!")
    # make sure style sheet is correct
    if style_sheet in plt.style.available:
        # set style of plot
        plt.style.use(style_sheet)
    else:
        # style is not correct
        raise ValueError("`style_sheet=%s` is not in the supported styles: %s" % (str(style_sheet),
                                                                                  str(plt.style.available)))
    # all linestyles
    linestyles = ['-', '--', '-.', ':']

    if use_linestyles is None:
        use_linestyles = ['-'] * len(dict_arrays)

    else:
        # check if linestyle is set right
        for use_linestyle in use_linestyles:
            if use_linestyle not in linestyles:
                # raise error
                raise ValueError("`linestyle=%s` is not in the styles: %s!" % (str(use_linestyle), str(linestyles)))

    # Check `points_value` type - it can be bool or list(bool)
    if isinstance(points_values, bool):
        # convert to list.
        points_values = [points_values] * len(dict_arrays)
    elif isinstance(points_values, list) and (len(points_values) != len(dict_arrays)):
        raise ValueError('`points_values` of type `list` must have same length as dictionary!')

    # single plot figure
    plt.subplot(1, 2, 1)
    for index, (use_label, array) in enumerate(dict_arrays.items()):
        # set steps plotted on x-axis - we can use step if 1 unit has different value
        if start_step > 0:
            # Offset all steps by start_step.
            steps = np.array(range(0, len(array))) * step_size + start_step
        else:
            steps = np.array(range(1, len(array) + 1)) * step_size
        # plot array as a single line
        plt.plot(steps, array, linestyle=use_linestyles[index], label=use_label)
        # Plots points values
        if points_values[index]:
            # Loop through each point and plot the label.
            for x, y in zip(steps, array):
                # Add text label to plo.
                plt.text(x, y, str(round(y, points_round)))
    # set title of figure
    plt.title(use_title)
    # set horizontal axis name
    plt.xlabel(use_xlabel)
    # set vertical axis name
    plt.ylabel(use_ylabel)
    # place legend best position
    plt.legend(loc='best') if use_label is not None else None
    # display grid depending on `use_grid`
    plt.grid(use_grid)
    # make figure nice
    plt.tight_layout()
    # get figure object from plot
    fig = plt.gcf()
    # get size of figure
    figsize = fig.get_size_inches()
    # change size depending on height and width variables
    figsize = [figsize[0] * width * magnify, figsize[1] * height * magnify]
    # set the new figure size with magnify
    fig.set_size_inches(figsize)
    # save figure to image if path is set
    fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None
    # show plot
    plt.show() if show_plot is True else None

    return


def plot_confusion_matrix(y_true, y_pred, use_title=None, classes='', normalize=False, style_sheet='ggplot',
                          cmap=plt.cm.Blues, verbose=0, width=3, height=1, magnify=1.2, use_dpi=50, path=None,
                          show_plot=True, **kwargs):
    r"""
    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    Arguments:

        y_true (:obj:`list / np.ndarray`):
            List of labels values.

        y_pred (:obj:`list / np.ndarray`):
            List of predicted label values.

        use_title (:obj:`int`, `optional`):
            Title on top of plot. This argument is optional and it will have a `None` value attributed
            inside the function.

        classes (:obj:`str`, `optional`, defaults to :obj:``):
            List of label names. This argument is optional and it has a default value attributed
            inside the function.

        normalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Normalize confusion matrix or not. This argument is optional and it has a default value attributed
            inside the function.

        style_sheet (:obj:`str`, `optional`, defaults to :obj:`ggplot`):
            Style of plot. Use plt.style.available to show all styles. This argument is optional and it has a default
            value attributed inside the function.

        cmap (:obj:`str`, `optional`, defaults to :obj:`plt.cm.Blues`):
            It is a plt.cm plot theme. Plot themes: `plt.cm.Blues`, `plt.cm.BuPu`, `plt.cm.GnBu`, `plt.cm.Greens`,
            `plt.cm.OrRd`. This argument is optional and it has a default value attributed inside the function.

        verbose (:obj:`int`, `optional`, defaults to :obj:`0`):
            To display confusion matrix value or not if set > 0. This argument is optional and it has a default
            value attributed inside the function.

        width (:obj:`int`, `optional`, defaults to :obj:`3`):
            Horizontal length of plot. This argument is optional and it has a default value attributed inside
            the function.

        height (:obj:`int`, `optional`, defaults to :obj:`1`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        magnify (:obj:`int`, `optional`, defaults to :obj:`1.2`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        use_dpi (:obj:`int`, `optional`, defaults to :obj:`50`):
            Vertical length of plot. This argument is optional and it has a default value attributed inside
            the function.

        path (:obj:`str`, `optional`):
            Vertical length of plot. This argument is optional and it will have a None value attributed inside
            the function.

        show_plot (:obj:`bool`, `optional`, defaults to :obj:`True`):
            if you want to call `plt.show()`. or not (if you run on a headless server). This argument is optional and
            it has a default value attributed inside the function.

        kwargs (:obj:`dict`, `optional`):
            Other arguments that might be deprecated or not included as details. This argument is optional and it will
            have a `None` value attributed inside the function.

    Returns:

        :obj:`np.ndarray`: Confusion matrix used to plot.

    Raises:

        DeprecationWarning: If arguments `title` is used.

        DeprecationWarning: If arguments `image` is used.

        DeprecationWarning: If arguments `dpi` is used.

        ValueError: If `y_true` and `y_pred` arrays don't have same length.

        ValueError: If `dict_arrays` doesn't have string keys.

        ValueError: If `dict_arrays` doesn't have array values.

        ValueError: If `style_sheet` is not valid.

    """

    # Handle deprecation warnings if `title` is used.
    if 'title' in kwargs:
        # assign same value
        use_title = kwargs['title']
        warnings.warn("`title` will be deprecated in future updates. Use `use_title` in stead!", DeprecationWarning)
    # Handle deprecation warnings if `image` is used.
    if 'image' in kwargs:
        # assign same value
        path = kwargs['image']
        warnings.warn("`image` will be deprecated in future updates. Use `path` in stead!", DeprecationWarning)
    # Handle deprecation warnings if `dpi` is used.
    if 'dpi' in kwargs:
        # assign same value
        use_dpi = kwargs['dpi']
        warnings.warn("`dpi` will be deprecated in future updates. Use `use_dpi` in stead!", DeprecationWarning)
    # Make sure labels have right format
    if len(y_true) != len(y_pred):
        # make sure lengths match
        raise ValueError("`y_true` needs to have same length as `y_pred`!")
    # make sure style sheet is correct
    if style_sheet in plt.style.available:
        # set style of plot
        plt.style.use(style_sheet)
    else:
        # style is not correct
        raise ValueError("`style_sheet=%s` is not in the supported styles: %s" % (str(style_sheet),
                                                                                  str(plt.style.available)))
    # Class labels setup. If none, generate from y_true y_pred
    classes = list(classes)
    if classes:
        assert len(set(y_true)) == len(classes)
    else:
        classes = set(y_true)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Nromalize setup
    if normalize is True:
        print("Normalized confusion matrix")
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        use_title = 'Normalized confusion matrix' if use_title is None else use_title
    else:
        print('Confusion matrix, without normalization')
        use_title = 'Confusion matrix, without normalization' if use_title is None else use_title
    # Print if verbose
    print(cm) if verbose > 0 else None
    # Plot setup
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=use_title,
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
    # Never display grid.
    plt.grid(False)
    # make figure nice
    plt.tight_layout()
    # get figure object from plot
    fig = plt.gcf()
    # get size of figure
    figsize = fig.get_size_inches()
    # change size depending on height and width variables
    figsize = [figsize[0] * width * magnify, figsize[1] * height * magnify]
    # set the new figure size with magnify
    fig.set_size_inches(figsize)
    # save figure to image if path is set
    fig.savefig(path, dpi=use_dpi, bbox_inches='tight') if path is not None else None
    # show plot
    plt.show() if show_plot is True else None

    return cm
