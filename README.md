# Machine Learning Things

[![Generic badge](https://img.shields.io/badge/Working-Progress-red.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Generic badge](https://img.shields.io/badge/Updated-Sep_2020-yellow.svg)]()
[![Generic badge](https://img.shields.io/badge/Website-Online-green.svg)]()

**Machine Learning Things (ml_things)** is a lightweight python library that contains functions and code snippets that 
I use in my everyday research with Machine Learning, Deep Learning, NLP.

I created this repo because I was tired of always looking up same code from older projects and I wanted to gain some experience in building a Python library. 
By making this available to everyone it gives me easy access to code I use frequently and it can help others in their machine learning work. 
If you find any bugs or something doesn't make sense please feel free to open an issue.

This library also contains Python code snippets that can speed up Machine Learning workflow.

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
|                 	|                                                                                                                                                                                                                          	|
|-----------------	|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| **Parameters:** 	| **variable_length_array** : array<br>Single arrays [1,2,3] or nested arrays [[1,2],[3]].<br><br>**fixed_length** : int<br>Max length of rows for numpy.<br><br>**axis** : int<br>Directions along rows: 1 or columns: 0. 	|
| **Returns:**    	| **numpy_array** :  <br>axis=1: fixed numpy array shape [len of array, fixed_length].                <br>axis=0: fixed numpy array shape [fixed_length, len of array].                                                    	|                                   	|

Example:

```python
>>> from ml_things import pad_array
>>> pad_array(variable_length_array=[[1,2],[3],[4,5,6]], fixed_length=5)
array([[1., 2., 0., 0., 0.],
       [3., 0., 0., 0., 0.],
       [4., 5., 6., 0., 0.]])
```

### batch_array [[source]](https://github.com/gmihaila/ml_things/blob/d18728fba08640d7f1bc060e299e4d4e84814a25/src/ml_things/array_functions.py#L98)

```python
def batch_array(list_values, batch_size)
```
