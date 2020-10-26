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

Pad variable length array to a fixed numpy array. It can handle single arrays [1,2,3] or nested arrays [[1,2],[3]].
    
By default it will padd zeros to the maximum length of row detected:

```python
>>> from ml_things import pad_array
>>> pad_array(variable_length_array=[[1,2],[3],[4,5,6]])
array([[1., 2., 0.],
       [3., 0., 0.],
       [4., 5., 6.]])
```

It can also pad to a custom size and with cusotm values:

```python
>>> pad_array(variable_length_array=[[1,2],[3],[4,5,6]], fixed_length=5, pad_value=99)
array([[ 1.,  2., 99., 99., 99.],
       [ 3., 99., 99., 99., 99.],
       [ 4.,  5.,  6., 99., 99.]])
```
       
### batch_array [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/array_functions.py#L98)



### plot_array [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/plot_functions.py#L22)



### plot_dict [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/plot_functions.py#L97)





### plot_confusion_matrix [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/plot_functions.py#L97)




### download_from [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/web_related.py#L21)




### clean_text [[source]](https://github.com/gmihaila/ml_things/blob/9ea16e6df75a907fadf8c40b29ef7b3da9d37701/src/ml_things/text_functions.py#L22)



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
