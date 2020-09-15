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
|**Parameters:**|**:param** <br>&nbsp;&nbsp; variable_length_array: Single arrays [1,2,3] or nested arrays [[1,2],[3]]. <br> **:param** <br>&nbsp;&nbsp; fixed_length: max length of rows for numpy. <br> **:param** <br>&nbsp;&nbsp; axis: directions along rows: 1 or columns: 0 |
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
|**Parameters:**|
**:param** <br>&nbsp;&nbsp; array: list of values. Can be of type list or np.ndarray. <br>
**:param** <br>&nbsp;&nbsp; step_size: steps shows on x-axis. Change if each steps is different than 1. <br>
**:param** <br>&nbsp;&nbsp; use_label: display label of values from array. <br>
**:param** <br>&nbsp;&nbsp; use_title: title on top of plot. <br>
**:param** <br>&nbsp;&nbsp; use_xlabel: horizontal axis label. <br>
**:param** <br>&nbsp;&nbsp; use_ylabel: vertical axis label. <br>
**:param** <br>&nbsp;&nbsp; style_sheet: style of plot. Use plt.style.available to show all styles. <br>
**:param** <br>&nbsp;&nbsp; use_grid: show grid on plot or not. <br>
**:param** <br>&nbsp;&nbsp; width: horizontal length of plot. <br>
**:param** <br>&nbsp;&nbsp; height: vertical length of plot. <br>
**:param** <br>&nbsp;&nbsp; use_linestyle: whtat style to use on line from ['-', '--', '-.', ':']. <br>
**:param** <br>&nbsp;&nbsp; use_dpi: quality of image saved from plot. 100 is prety high. <br>
**:param** <br>&nbsp;&nbsp; path: path where to save the plot as an image - if set to None no image will be saved. <br>
**:param** <br>&nbsp;&nbsp; show_plot: if you want to call `plt.show()`. or not (if you run on a headless server).|
|**Returns:**||



### plot_confusion_matrix [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/plot_functions.py#L97)

```python
plot_confusion_matrix(y_true, y_pred, classes='', normalize=False, title=None, cmap=plt.cm.Blues, image=None,
                          verbose=0, magnify=1.2, dpi=50)
```

|Description:|This function prints and plots the confusion matrix.<br>Normalization can be applied by setting `normalize=True`.
<br>y_true needs to contain all possible labels.|
|:-|:-|
|**Parameters:**|**:param** <br>&nbsp;&nbsp; y_true: array labels values. <br>**:param** <br>&nbsp;&nbsp; y_pred: array predicted label values.
**:param** <br>&nbsp;&nbsp; classes: array list of label names. <br>
**:param** <br>&nbsp;&nbsp; normalize: bool normalize confusion matrix or not. <br>
**:param** <br>&nbsp;&nbsp; title: str string title of plot. <br>
**:param** <br>&nbsp;&nbsp; cmap: plt.cm plot theme. <br>
**:param** <br>&nbsp;&nbsp; image: str path to save plot in an image. <br>
**:param** <br>&nbsp;&nbsp; verbose: int print confusion matrix when calling function. <br>
**:param** <br>&nbsp;&nbsp; magnify: int zoom of plot. <br>
**:param** <br>&nbsp;&nbsp; dpi: int clarity of plot.|
|**Returns:**||


### download_from [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/web_related.py#L21)

```python
download_from(url, path)
```
|Description:|Download file from url.|
|:-|:-|
|**Parameters:**|
**:param url:** <br>&nbsp;&nbsp;  web path of file. <br>
**:param path:** <br>&nbsp;&nbsp;  path to save the file.|
|**Returns:**|**:return:** <br>&nbsp;&nbsp; path where file was saved|



# Snippets

This is a very large variety of Python snippets without a certain theme. I put them in the most frequently used ones while keeping a logical order.
I like to have them as simple and as efficient as possible.

| Name | Description |
|:-|:-|
| Read FIle     	| One liner to read any file.
| Write File 	       | One liner to write a string to a file.
| Debug         	| Start debugging after this line.
| Pip Install GitHub	| Install library directly from GitHub using `pip`.
| Parse Argument     | Parse arguments given when running a `.py` file.
| Using Doctest      | How to run a simple unittesc using function documentaiton. Useful when need to do unittest inside notebook.
| Unittesting        | Simple example of creating unittests.
| Sort Keys          | Sorting dicitonary using key values.
| Sort Values        | Sorting dicitonary using values.


# Notebooks

This is where I keep notebooks of some previous projects which I turnned them into small tutorials. A lot of times I use them as basis for starting a new project.

All of the notebooks are in **Google Colab**. Never heard of Google Colab? :scream_cat: You have to check out the [Overview of Colaboratory](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiD1aD06trrAhVRXK0KHRC4DgQQjBAwBHoECAYQBA&url=https%3A%2F%2Fcolab.research.google.com%2Fnotebooks%2Fbasic_features_overview.ipynb&usg=AOvVaw0gXOkR6JGGFlwsxrkuYm7F), [Introduction to Colab and Python](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiD1aD06trrAhVRXK0KHRC4DgQQjBAwA3oECAYQCg&url=https%3A%2F%2Fcolab.research.google.com%2Fgithub%2Ftensorflow%2Fexamples%2Fblob%2Fmaster%2Fcourses%2Fudacity_intro_to_tensorflow_for_deep_learning%2Fl01c01_introduction_to_colab_and_python.ipynb&usg=AOvVaw2pr-crqP30RHfDs7hjKNnc) and what I think is a great medium article about it [to configure Google Colab Like a Pro](https://medium.com/@robertbracco1/configuring-google-colab-like-a-pro-d61c253f7573).

If you check the `/ml_things/notebooks/` a lot of them are not listed here because they are not in a 'polished' form yet. These are the notebooks that are good enough to share with everyone:

| Name 	| Description 	| Colab Link 	|
|:- |:- |:- |
| Pretrain Transformers     | Simple notebook to pretrain transformers model on a specific dataset using [transformers]() from Huggingface |            	|
|      	|             	|            	|
|      	|             	|            	|
|      	|             	|            	|


# Final Note

Thank you for checking out my repo. I am a perfectionist so I will do a lot of changes when it comes to small details. 

Lern more about me? Check out my website **[gmihaila.github.io](http://gmihaila.github.io)**!
