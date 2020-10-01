# Machine Learning Things

[![Generic badge](https://img.shields.io/badge/Working-Progress-red.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Generic badge](https://img.shields.io/badge/Updated-Sep_2020-yellow.svg)]()
[![Generic badge](https://img.shields.io/badge/Website-Online-green.svg)]()

**Machine Learning Things** is a lightweight python library that contains functions and code snippets that 
I use in my everyday research with Machine Learning, Deep Learning, NLP.

I created this repo because I was tired of always looking up same code from older projects and I wanted to gain some experience in building a Python library. 
By making this available to everyone it gives me easy access to code I use frequently and it can help others in their machine learning work. 
If you find any bugs or something doesn't make sense please feel free to open an issue.

That is not all! This library also contains Python code snippets and notebooks that speed up my Machine Learning workflow.

# Table of contents

* **[ML_things](https://github.com/gmihaila/ml_things#ml_things)**: Details on the ml_things libary how to install and use it.

* **[Snippets](https://github.com/gmihaila/ml_things#snippets)**: Curated list of Python snippets I frequently use.

* **[Comments](https://github.com/gmihaila/ml_things#comments)**: Some small snippets of how I like to comment my code.

* **[Notebooks](https://github.com/gmihaila/ml_things#notebooks)**: Google Colab Notebooks from old project that I converted to tutorials.

* **[Final Note](https://github.com/gmihaila/ml_things#final-note)**

<br/>

# ML_things

## Installation

This repo is tested with Python 3.6+.

It's always good practice to install `ml_things` in a [virtual environment](https://docs.python.org/3/library/venv.html). If you guidance on using Python's virtual environments you can check out the user guide [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

You can install `ml_things` with pip from GitHub:

```bash
pip install git+https://github.com/gmihaila/ml_things
```

## Functions

### pad_array [[source]](https://github.com/gmihaila/ml_things/blob/d18728fba08640d7f1bc060e299e4d4e84814a25/src/ml_things/array_functions.py#L21)

```python
def pad_array(variable_length_array, fixed_length=None, axis=1)
```
|Description:|Pad variable length array to a fixed numpy array. <br>It can handle single arrays [1,2,3] or nested arrays [[1,2],[3]].|
|:-|:-|
|**Parameters:**|**:param** <br>&nbsp;&nbsp; variable_length_array: Single arrays [1,2,3] or nested arrays [[1,2],[3]]. <br> **:param** <br>&nbsp;&nbsp; fixed_length: max length of rows for numpy. <br> **:param** <br>&nbsp;&nbsp; axis: directions along rows: 1 or columns: 0<br> **:param** <br>&nbsp;&nbsp; pad_value: what value to use as padding, default is 0. |
|**Returns:**|**:return:** <br>&nbsp;&nbsp; numpy_array: <br>&nbsp;&nbsp;&nbsp;&nbsp; axis=1: fixed numpy array shape [len of array, fixed_length]. <br>&nbsp;&nbsp;&nbsp;&nbsp; axis=0: fixed numpy array shape [fixed_length, len of array].|                                                                                                                                 


Example:

```python
>>> from ml_things import pad_array
>>> pad_array(variable_length_array=[[1,2],[3],[4,5,6]], fixed_length=5)
array([[1., 2., 0., 0., 0.],
       [3., 0., 0., 0., 0.],
       [4., 5., 6., 0., 0.]])
```

### batch_array [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/array_functions.py#L98)

```python
def batch_array(list_values, batch_size)
```

|Description:|Split a list into batches/chunks.<br> Last batch size is remaining of list values.|
|:-|:-|
|**Parameters:**|**:param** <br>&nbsp;&nbsp; list_values: can be any kind of list/array.<br> **:param** <br>&nbsp;&nbsp; batch_size: int value of the batch length.|
|**Returns:**|**:return:** <br>&nbsp;&nbsp; List of batches from list_values.|

### plot_array [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/plot_functions.py#L22)

```python
plot_array(array, step_size=1, use_label=None, use_title=None, use_xlabel=None, use_ylabel=None,
               style_sheet='ggplot', use_grid=True, width=3, height=1, use_linestyle='-', use_dpi=20, path=None,
               show_plot=True)
```

|Description:|Create plot from a single array of values.|
|:-|:-|
|**Parameters:**|**:param** <br>&nbsp;&nbsp; array: list of values. Can be of type list or np.ndarray. <br>**:param** <br>&nbsp;&nbsp; step_size: steps shows on x-axis. Change if each steps is different than 1. <br>**:param** <br>&nbsp;&nbsp; use_label: display label of values from array. <br>**:param** <br>&nbsp;&nbsp; use_title: title on top of plot. <br>**:param** <br>&nbsp;&nbsp; use_xlabel: horizontal axis label. <br>**:param** <br>&nbsp;&nbsp; use_ylabel: vertical axis label. <br>**:param** <br>&nbsp;&nbsp; style_sheet: style of plot. Use plt.style.available to show all styles. <br>**:param** <br>&nbsp;&nbsp; use_grid: show grid on plot or not. <br>**:param** <br>&nbsp;&nbsp; width: horizontal length of plot. <br>**:param** <br>&nbsp;&nbsp; height: vertical length of plot. <br>**:param** <br>&nbsp;&nbsp; use_linestyle: whtat style to use on line from ['-', '--', '-.', ':']. <br>**:param** <br>&nbsp;&nbsp; use_dpi: quality of image saved from plot. 100 is prety high. <br>**:param** <br>&nbsp;&nbsp; path: path where to save the plot as an image - if set to None no image will be saved. <br>**:param** <br>&nbsp;&nbsp; show_plot: if you want to call `plt.show()`. or not (if you run on a headless server).|
|**Returns:**||


### plot_dict [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/plot_functions.py#L97)

```python
plot_dict(dict_arrays, step_size=1, use_title=None, use_xlabel=None, use_ylabel=None,
              style_sheet='ggplot', use_grid=True, width=3, height=1, use_linestyles=None, use_dpi=20, path=None,
              show_plot=True)
```

|Description:|Create plot from a dictionary of lists.|
|:-|:-|
|**Parameters:**|**:param** <br>&nbsp;&nbsp; dict_arrays: dictionary of lists or np.array <br> **:param** <br>&nbsp;&nbsp; step_size: steps shows on x-axis. Change if each steps is different than 1. <br> **:param** <br>&nbsp;&nbsp; use_title: title on top of plot. <br> **:param** <br>&nbsp;&nbsp; use_xlabel: horizontal axis label. <br> **:param** <br>&nbsp;&nbsp; use_ylabel: vertical axis label. <br> **:param** <br>&nbsp;&nbsp; style_sheet: style of plot. Use plt.style.available to show all styles. <br> **:param** <br>&nbsp;&nbsp; use_grid: show grid on plot or not. <br> **:param** <br>&nbsp;&nbsp; width: horizontal length of plot. <br> **:param** <br>&nbsp;&nbsp; height: vertical length of plot. <br> **:param** <br>&nbsp;&nbsp; use_linestyles: array of styles to use on line from ['-', '--', '-.', ':']. <br> **:param** <br>&nbsp;&nbsp; use_dpi: quality of image saved from plot. 100 is pretty high. <br> **:param** <br>&nbsp;&nbsp; path: path where to save the plot as an image - if set to None no image will be saved. <br> **:param** <br>&nbsp;&nbsp; show_plot: if you want to call `plt.show()`. or not (if you run on a headless server).|
|**Returns:**||



### plot_confusion_matrix [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/plot_functions.py#L97)

```python
plot_confusion_matrix(y_true, y_pred, classes='', normalize=False, title=None, cmap=plt.cm.Blues, image=None,
                          verbose=0, magnify=1.2, dpi=50)
```

| Description: 	| This function prints and plots the confusion matrix.<br>Normalization can be applied by setting normalize=True. <br>y_true needs to contain all possible labels. 	|
|:-	|:-	|
| **Parameters:** 	| **:param** <br>   y_true: array labels values. <br>**:param** <br>   y_pred: array predicted label values.**:param**<br>   classes: array list of label names. <br>**:param** <br>   normalize: bool normalize confusion matrix or not. <br>**:param** <br>   title: str string title of plot. <br>**:param** <br>   cmap: plt.cm plot theme. <br>**:param** <br>   image: str path to save plot in an image. <br>**:param** <br>   verbose: int print confusion matrix when calling function. <br>**:param** <br>   magnify: int zoom of plot. <br>**:param** <br>   dpi: int clarity of plot. 	|
| **Returns:** 	|  	|



### download_from [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/web_related.py#L21)

```python
download_from(url, path)
```
|Description:|Download file from url.|
|:-|:-|
|**Parameters:**|**:param** <br>&nbsp;&nbsp; url: web path of file. <br>**:param** <br>&nbsp;&nbsp;  path: path to save the file.|
|**Returns:**|**:return:** <br>&nbsp;&nbsp; path where file was saved|



### clean_text [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/text_functions.py#L22)

```python
clean_text(text, full_clean=False, punctuation=False, numbers=False, lower=False, extra_spaces=False,
               control_characters=False, tokenize_whitespace=False, remove_characters='')
```


|Description:|Clean text using various techniques.|
|:-|:-|
|**Parameters:**|**:param** <br>&nbsp;&nbsp;  text: string that needs cleaning. <br>**:param** <br>&nbsp;&nbsp;  full_clean: remove: punctuation, numbers, extra space, control characters and lower case. <br>**:param** <br>&nbsp;&nbsp;  punctuation: remove punctuation from text. <br>**:param** <br>&nbsp;&nbsp;  numbers: remove digits from text. <br>**:param** <br>&nbsp;&nbsp;  lower: lower case all text. <br>**:param** <br>&nbsp;&nbsp;  extra_spaces: remove extra spaces - everything beyond one space. <br>**:param** <br>&nbsp;&nbsp;  control_characters: remove characters like `\n`, `\t` etc. <br>**:param** <br>&nbsp;&nbsp;  tokenize_whitespace: return a list of tokens split on whitespace. <br>**:param** <br>&nbsp;&nbsp;  remove_characters: remove defined characters form text. <br>|
|**Returns:**|**:return:** <br>&nbsp;&nbsp; cleaned text or list of tokens of cleaned text.|


# Snippets

This is a very large variety of Python snippets without a certain theme. I put them in the most frequently used ones while keeping a logical order.
I like to have them as simple and as efficient as possible.

| Name | Description |
|:-|:-|
| [Read FIle](https://gmihaila.github.io/useful/useful/#read-file)     	| One liner to read any file.
| [Write File](https://gmihaila.github.io/useful/useful/#write-file) 	       | One liner to write a string to a file.
| [Debug](https://gmihaila.github.io/useful/useful/#debug)         	| Start debugging after this line.
| [Pip Install GitHub](https://gmihaila.github.io/useful/useful/#pip-install-github)	| Install library directly from GitHub using `pip`.
| [Parse Argument](https://gmihaila.github.io/useful/useful/#parse-argument)     | Parse arguments given when running a `.py` file.
| [Doctest](https://gmihaila.github.io/useful/useful/#doctest)      | How to run a simple unittesc using function documentaiton. Useful when need to do unittest inside notebook.
| [Fix Text](https://gmihaila.github.io/useful/useful/#fix-text) | Since text data is always messy, I always use it. It is great in fixing any bad Unicode.
| [Current Date](https://gmihaila.github.io/useful/useful/#current-date)     | How to get current date in Python. I use this when need to name log files.
| [Current Time](https://gmihaila.github.io/useful/useful/#current-time) | Get current time in Python.
| [Remove Punctuation](https://gmihaila.github.io/useful/useful/#remove-punctuation)        | The fastest way to remove punctuation in Python3.
| [PyTorch-Dataset](https://gmihaila.github.io/useful/useful/#dataset)       | Code sample on how to create a PyTorch Dataset.
| [PyTorch-Device](https://gmihaila.github.io/useful/useful/#pytorch-device)        | How to setup device in PyTorch to detect if GPU is available.


# Comments

These are a few snippets of how I like to comment my code. I saw a lot of different ways of how people comment their code. One thing is for sure: *any comment is better than no comment*.

I try to follow as much as I can the [PEP 8 — the Style Guide for Python Code](https://pep8.org/#code-lay-out).

When I comment a function or class:
```python
# required import for variables type declaration
from typing import List, Optional, Tuple, Dict

def my_function(function_argument: str, another_argument: Optional[List[int]] = None,
                another_argument_: bool = True) -> Dict[str, int]
       """Function/Class main comment. 

       More details with enough spacing to make it easy to follow.

       Arguments:
              function_argument (:obj:`str`):
                     A function argument description.
              another_argument (:obj:`List[int]`, `optional`):
                     This argument is optional and it will have a None value attributed inside the function.
              another_argument_ (:obj:`bool`, `optional`, defaults to :obj:`True`):
                     This argument is optional and it has a default value.
                     The variable name has `_` to avoid conflict with similar name.

       Returns:
              :obj:`Dict[str: int]`: The function returns a dicitonary with string keys and int values.
                     A class will not have a return of course.

       """
       
       # make sure we keep out promise and return the variable type we described.
       return {'argument': function_argument}
```


# Notebooks

This is where I keep notebooks of some previous projects which I turnned them into small tutorials. A lot of times I use them as basis for starting a new project.

All of the notebooks are in **Google Colab**. Never heard of Google Colab? :scream_cat: You have to check out the [Overview of Colaboratory](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiD1aD06trrAhVRXK0KHRC4DgQQjBAwBHoECAYQBA&url=https%3A%2F%2Fcolab.research.google.com%2Fnotebooks%2Fbasic_features_overview.ipynb&usg=AOvVaw0gXOkR6JGGFlwsxrkuYm7F), [Introduction to Colab and Python](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiD1aD06trrAhVRXK0KHRC4DgQQjBAwA3oECAYQCg&url=https%3A%2F%2Fcolab.research.google.com%2Fgithub%2Ftensorflow%2Fexamples%2Fblob%2Fmaster%2Fcourses%2Fudacity_intro_to_tensorflow_for_deep_learning%2Fl01c01_introduction_to_colab_and_python.ipynb&usg=AOvVaw2pr-crqP30RHfDs7hjKNnc) and what I think is a great medium article about it [to configure Google Colab Like a Pro](https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573).

If you check the `/ml_things/notebooks/` a lot of them are not listed here because they are not in a 'polished' form yet. These are the notebooks that are good enough to share with everyone:

| Name 	| Description 	| Google Colab 	|
|:- |:- |:- |
| [PyTorchText](https://gmihaila.github.io/tutorial_notebooks/pytorchtext/) | This notebook is an example of using pytorchtext powerful BucketIterator function which allows grouping examples of similar lengths to provide the most optimal batching method. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/pytorchtext.ipynb) |
| [Pretrain Transformers](https://gmihaila.github.io/tutorial_notebooks/pretrain_transformer/)     | This notebook is used to pretrain transformers models using Huggingface. |      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14KCDms4YLrE7Ekxl9VtrdT229UTDyim3#offline=true&sandboxMode=true)|
|      	|             	|            	|
|      	|             	|            	|
|      	|             	|            	|


# Final Note

Thank you for checking out my repo. I am a perfectionist so I will do a lot of changes when it comes to small details. 

Lern more about me? Check out my website **[gmihaila.github.io](http://gmihaila.github.io)**
