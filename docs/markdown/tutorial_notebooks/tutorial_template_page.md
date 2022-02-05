# **:gear: Title**  

## **:construction: Work in progress :construction_worker:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/finetune_transformers_pytorch.ipynb) &nbsp;
[![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/finetune_transformers_pytorch.ipynb)
[![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/h13v19ns3oig2rl/finetune_transformers_pytorch.ipynb?dl=1)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## **Info**

Intro to this tutorial

<br>

## **What should I know for this notebook?**

Any requirements.

<br>

## **How to use this notebook?**

Instructions.

<br>

## **What <any specific question>?**

Tutorial specific answer.

<br>

## **Dataset**

I will use the well known movies reviews positive - negative labeled [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The description provided on the Stanford website:

*This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details.*

**Why this dataset?** I believe is an easy to understand and use dataset for classification. I think sentiment data is always fun to work with.

<br>

## **Coding**

Now let's do some coding! We will go through each coding cell in the notebook and describe what it does, what's the code, and when is relevant - show the output

I made this format to be easy to follow if you decide to run each code cell in your own python notebook.

When I learn from a tutorial I always try to replicate the results. I believe it's easy to follow along if you have the code next to the explanations.

<br>

### **Downloads**

Download the *Large Movie Review Dataset* and unzip it locally.

**Code Cell:**
```shell
# download the dataset
!wget -q -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# unzip it
!tar -zxf /content/aclImdb_v1.tar.gz
```

<br>

### **Installs**

* **[transformers](https://github.com/huggingface/transformers)** library needs to be installed to use all the awesome code from Hugging Face. To get the latest version I will install it straight from GitHub.

* **[ml_things](https://github.com/gmihaila/ml_things)** library used for various machine learning related tasks. I created this library to reduce the amount of code I need to write for each machine learning project. Give it a try!

**Code Cell:**
```shell
# Install transformers library.
!pip install -q git+https://github.com/huggingface/transformers.git
# Install helper functions.
!pip install -q git+https://github.com/gmihaila/ml_things.git
```
**Output:**
```shell
Installing build dependencies ... done
Getting requirements to build wheel ... done
Preparing wheel metadata ... done
 |████████████████████████████████| 2.9MB 6.7MB/s 
 |████████████████████████████████| 890kB 48.9MB/s 
 |████████████████████████████████| 1.1MB 49.0MB/s 
Building wheel for transformers (PEP 517) ... done
Building wheel for sacremoses (setup.py) ... done
 |████████████████████████████████| 71kB 5.2MB/s 
Building wheel for ml-things (setup.py) ... done
Building wheel for ftfy (setup.py) ... done
```

<br>

### **Imports**

Import all needed libraries for this notebook.

Declare parameters used for this notebook:
*
*

**Code Cell:**
```python

```

<br>

### **Helper Functions**


**Class() / function()** 

Class / function description.

**Code Cell:**
```python

```

<br>

### **Load Model and Tokenizer**

Loading the three essential parts of the pretrained transformers: *configuration*, *tokenizer* and *model*. I also need to load the
model on the device I'm planning to use (GPU / CPU).


**Code Cell:**
```python

```
**Output:**
```shell

```


<br>

### **Dataset and DataLoader**

Details.


**Code Cell:**
```python

```
**Output:**
```shell

```


<br>

### **Train**


**Code Cell:**
```python

```
**Output:**
```shell

```

Use ColabImage plots straight in here
![]()

<br>

### **Evaluate**

Evaluation!

**Code Cell:**
```python

```
**Output:**
```shell

```

Use ColabImage plots straight in here
![]()

<br>

## **Final Note**

If you made it this far Congrats :confetti_ball: and Thank you :pray: for your interest in my tutorial!

Other details.

If you see something wrong please let me know by opening an 
**[issue on my ml_things](https://github.com/gmihaila/ml_things/issues/new/choose)** GitHub repository! 

A lot of tutorials out there are mostly a one-time thing and are not being maintained. I plan on keeping my 
tutorials up to date as much as I can.

<br>

## **Contact :fishing_pole_and_fish:**

:cat: GitHub: [gmihaila](https://github.com/gmihaila){:target="_blank"}

:earth_americas: Website: [gmihaila.github.io](https://gmihaila.github.io)

:necktie: LinkedIn: [mihailageorge](https://www.linkedin.com/in/mihailageorge)

:mailbox_with_mail: Email: [georgemihaila@my.unt.edu.com](mailto:georgemihaila@my.unt.edu.com?subject=GitHub%20Website)

:busts_in_silhouette: Schedule meeting: [calendly.com/georgemihaila](https://calendly.com/georgemihaila)

<br>

**Thank you!**

<br>


**Find out more [About Me](https://gist.github.com/gmihaila/b8f5bdd93e577060d17048fd6a24b39d).**


<br>