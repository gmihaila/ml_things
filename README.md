# Machine Learning Things

[![Generic badge](https://img.shields.io/badge/Working-Progress-red.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Generic badge](https://img.shields.io/badge/Updated-Oct_2020-yellow.svg)]()
[![Generic badge](https://img.shields.io/badge/Website-Online-green.svg)](https://gmihaila.github.io)

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

Create plot from a single array of values.

```python
plot_array([1,3,5,3,7,5,8,10], path='img.png', magnify=0.5, use_title='A Random Plot', start_step=0.3, step_size=0.1, points_values=True)
```
![plot_array](data:image/png;base64,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)


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
       r"""Function/Class main comment. 

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
