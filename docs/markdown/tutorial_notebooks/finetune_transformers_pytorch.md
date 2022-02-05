# **:violin: Fine-tune Transformers in PyTorch using Hugging Face Transformers**  

## **Complete tutorial on how to fine-tune 73 transformer models for text classification — no code changes necessary!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/finetune_transformers_pytorch.ipynb) &nbsp;
[![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/finetune_transformers_pytorch.ipynb)
[![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/tsqicfqgt8v87ae/finetune_transformers_pytorch.ipynb?dl=1)
[![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://medium.com/@gmihaila/fine-tune-transformers-in-pytorch-using-transformers-57b40450635)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## **Info**

This notebook is designed to use a pretrained transformers model and fine-tune it on a classification task. 
The focus of this tutorial will be on the code itself and how to adjust it to your needs.

This notebook is using the [AutoClasses](https://huggingface.co/transformers/model_doc/auto.html) from 
[transformer](https://github.com/huggingface/transformers) by [Hugging Face](https://huggingface.co/)  functionality. 
This functionality can guess a model's configuration, tokenizer and architecture just by passing in the model's name. 
This allows for code reusability on a large number of transformers models!

<br>

## **What should I know for this notebook?**

I provided enough instructions and comments to be able to follow along with minimum Python coding knowledge.

Since I am using PyTorch to fine-tune our transformers models any knowledge on PyTorch is very useful. Knowing a little 
bit about the [transformers](https://github.com/huggingface/transformers) library helps too.

<br>

## **How to use this notebook?**

I built this notebook with reusability in mind. The way I load the dataset into the PyTorch Dataset class is pretty standard and can be easily reused for any other dataset.

The only modifications needed to use your own dataset will be in reading in the dataset inside the **MovieReviewsDataset** class which uses PyTorch **Dataset**. The **DataLoader** will return a dictionary of batch inputs format so that it can be fed straight to the model using the statement: `outputs = model(**batch)`. *As long as this statement holds, the rest of the code will work!*

<br>

## **What transformers models work with this notebook?**

There are rare cases where I use a different model than Bert when dealing with classification from text data. When there is a need to run a different transformer model architecture, which one would work with this code?

Since the name of the notebooks is *finetune_transformers* it should work with more than one type of transformers.

I ran this notebook across [all the pretrained models](https://huggingface.co/transformers/pretrained_models.html#pretrained-models) found on Hugging Face Transformer. This way you know ahead of time if the model you plan to use works with this code without any modifications.


The list of pretrained transformers models that work with this notebook can be found 
[here](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/finetune_transformers_pytorch_status_models.md#pretrained-models-that-work-with-finetune_transformers_pytorchipynb). 
There are **73 models that worked** :smile: and **33 models that failed to work** :cry: with this notebook.


<br>

## **Dataset**

This notebook will cover fine-tune transformers for binary classification task. I will use the well known movies reviews positive - negative labeled [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The description provided on the Stanford website:

*This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details.*

**Why this dataset?** I believe is an easy to understand and use dataset for classification. I think sentiment data is always fun to work with.

<br>

## **Coding**

Now let's do some coding! We will go through each coding cell in the notebook and describe what it does, what's the code, and when is relevant - show the output

I made this format to be easy to follow if you decide to run each code cell in your own python notebook.

When I learn from a tutorial I always try to replicate the results. I believe it's easy to follow along if you have the code next to the explanations.

<br>

## **Downloads**

Download the *Large Movie Review Dataset* and unzip it locally.

**Code Cell:**
```shell
# download the dataset
!wget -q -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# unzip it
!tar -zxf /content/aclImdb_v1.tar.gz
```

<br>

## **Installs**

* **[transformers](https://github.com/huggingface/transformers)** library needs to be installed to use all the awesome code from Hugging Face. To get the latest version I will install it straight from GitHub.

* **[ml_things](https://github.com/gmihaila/ml_things)** library used for various machine learning related tasks. I created this library to reduce the amount of code I need to write for each machine learning project. Give it a try!

```shell
# Install transformers library.
!pip install -q git+https://github.com/huggingface/transformers.git
# Install helper functions.
!pip install -q git+https://github.com/gmihaila/ml_things.git
```

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


<br>

## **Imports**

Import all needed libraries for this notebook.

Declare parameters used for this notebook:

* `set_seed(123)` - Always good to set a fixed seed for reproducibility.
* `epochs` - Number of training epochs (authors recommend between 2 and 4).
* `batch_size` - Number of batches - depending on the max sequence length and GPU memory. For 512 sequence length a batch of 10 USUALY works without cuda memory issues. For small sequence length can try batch of 32 or higher.
* `max_length` - Pad or truncate text sequences to a specific length. I will set it to 60 to speed up training.
* `device` - Look for gpu to use. Will use `cpu` by default if no `gpu` found.
* `model_name_or_path` - Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk. I always like to start off with `bert-base-cased`: *12-layer, 768-hidden, 12-heads, 109M parameters. Trained on cased English text.*
* `labels_ids` - Dictionary of labels and their id - this will be used to convert string labels to numbers.
* `n_labels` - How many labels are we using in this dataset. This is used to decide size of classification head.


```python
import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (AutoConfig, 
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )


# Set seed for reproducibility,
set_seed(123)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batch_size = 32

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 60

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.
model_name_or_path = 'bert-base-cased'

# Dicitonary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'neg': 0, 'pos': 1}

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)
```

<br>

## **Helper Functions**

I like to keep all Classes and functions that will be used in this notebook under this section to help maintain a clean look of the notebook:

**MovieReviewsDataset(Dataset)** 

If you worked with PyTorch before this is pretty standard. We need this class to read in our dataset, parse it, use 
tokenizer that transforms text into numbers and get it into a nice format to be fed to the model.

Lucky for use, Hugging Face thought of everything and made the **tokenizer** do all the heavy lifting (split text into 
tokens, padding, truncating, encode text into numbers) and is very easy to use!

In this class I only need to read in the content of each file, use **[fix_text](https://pypi.org/project/ftfy/)** to fix any Unicode problems and keep track 
of positive and negative sentiments.

I will append all texts and labels in lists that later I will feed to the tokenizer and to the label ids to transform everything 
into numbers.

There are three main parts of this PyTorch **Dataset** class:

* **__init__()** where we read in the dataset and transform text and labels into numbers.

* **__len__()** where we need to return the number of examples we read in. This is used when calling 
**len(MovieReviewsDataset())**.

* **__getitem()__** always takes as an input an int value that represents which example from our examples to return from our 
dataset. If a value of 3 is passed, we will return the example form our dataset at position 3. It needs to return an 
object with the format that can be fed to our model. Luckily our tokenizer does that for us and returns a dictionary 
of variables ready to be fed to the model in this way: `model(**inputs)`.


```python
class MovieReviewsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens and where the text gets encoded using
  loaded tokenizer.

  This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.
    
    use_tokenizer (:obj:`transformers.tokenization_?`):
        Transformer type tokenizer used to process raw text into numbers.

    labels_ids (:obj:`dict`):
        Dictionary to encode any labels names into numbers. Keys map to 
        labels names and Values map to number associated to those labels.

    max_sequence_len (:obj:`int`, `optional`)
        Value to indicate the maximum desired sequence to truncate or pad text
        sequences. If no value is passed it will used maximum sequence size
        supported by the tokenizer and model.

  """

  def __init__(self, path, use_tokenizer, labels_ids, max_sequence_len=None):

    # Check if path exists.
    if not os.path.isdir(path):
      # Raise error if path is invalid.
      raise ValueError('Invalid `path` variable! Needs to be a directory')
    # Check max sequence length.
    max_sequence_len = use_tokenizer.max_len if max_sequence_len is None else max_sequence_len
    texts = []
    labels = []
    print('Reading partitions...')
    # Since the labels are defined by folders with data we loop 
    # through each label.
    for label, label_id,  in tqdm(labels_ids.items()):
      sentiment_path = os.path.join(path, label)

      # Get all files from path.
      files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
      print('Reading %s files...' % label)
      # Go through each file and read its content.
      for file_name in tqdm(files_names):
        file_path = os.path.join(sentiment_path, file_name)

        # Read content.
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        # Fix any unicode issues.
        content = fix_text(content)
        # Save content.
        texts.append(content)
        # Save encode labels.
        labels.append(label_id)

    # Number of exmaples.
    self.n_examples = len(labels)
    # Use tokenizer on texts. This can take a while.
    print('Using tokenizer on all texts. This can take a while...')
    self.inputs = use_tokenizer(texts, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt',  max_length=max_sequence_len)
    # Get maximum sequence length.
    self.sequence_len = self.inputs['input_ids'].shape[-1]
    print('Texts padded or truncated to %d length!' % self.sequence_len)
    # Add labels.
    self.inputs.update({'labels':torch.tensor(labels)})
    print('Finished!\n')

    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return self.n_examples

  def __getitem__(self, item):
    r"""Given an index return an example from the position.
    
    Arguments:

      item (:obj:`int`):
          Index position to pick an example to return.

    Returns:
      :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
      It holddes the statement `model(**Returned Dictionary)`.

    """

    return {key: self.inputs[key][item] for key in self.inputs.keys()}
```

**train(dataloader, optimizer_, scheduler_, device_)**

I created this function to perform a full pass through the **DataLoader** object (the **DataLoader** object is created 
from our **Dataset* type object using the **MovieReviewsDataset** class). This is basically one epoch train through the entire dataset.

The **dataloader** is created from PyTorch DataLoader which takes the object created from **MovieReviewsDataset** class and 
puts each example in batches. This way we can feed our model batches of data!

The **optimizer_** and **scheduler_** are very common in PyTorch. They are required to update the parameters of our model 
and update our learning rate during training. There is a lot more than that but I won't go into details. 
*This can actually be a huge rabbit hole since A LOT happens behind these functions that we don't need to worry. Thank you PyTorch!*

In the process we keep track of the actual labels and the predicted labels along with the loss.

```python
def train(dataloader, optimizer_, scheduler_, device_):
  r"""
  Train pytorch model on a single pass through the data loader.

  It will use the global variable `model` which is the transformer model 
  loaded on `_device` that we want to train on.

  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.

      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

  Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss].
  """

  # Use global variable for model.
  global model

  # Tracking variables.
  predictions_labels = []
  true_labels = []
  # Total loss for this epoch.
  total_loss = 0

  # Put the model into training mode.
  model.train()

  # For each batch of training data...
  for batch in tqdm(dataloader, total=len(dataloader)):

    # Add original labels - use later for evaluation.
    true_labels += batch['labels'].numpy().flatten().tolist()
    
    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
    
    # Always clear any previously calculated gradients before performing a
    # backward pass.
    model.zero_grad()

    # Perform a forward pass (evaluate the model on this training batch).
    # This will return the loss (rather than the model output) because we
    # have provided the `labels`.
    # The documentation for this a bert model function is here: 
    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    outputs = model(**batch)

    # The call to `model` always returns a tuple, so we need to pull the 
    # loss value out of the tuple along with the logits. We will use logits
    # later to calculate training accuracy.
    loss, logits = outputs[:2]

    # Accumulate the training loss over all of the batches so that we can
    # calculate the average loss at the end. `loss` is a Tensor containing a
    # single value; the `.item()` function just returns the Python value 
    # from the tensor.
    total_loss += loss.item()

    # Perform a backward pass to calculate the gradients.
    loss.backward()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update parameters and take a step using the computed gradient.
    # The optimizer dictates the "update rule"--how the parameters are
    # modified based on their gradients, the learning rate, etc.
    optimizer.step()

    # Update the learning rate.
    scheduler.step()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Convert these logits to list of predicted labels values.
    predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)
  
  # Return all true labels and prediction for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss
```

**validation(dataloader, device_)**

I implemented this function in a very similar way as **train** but without the parameters update, backward pass and 
gradient decent part. We don't need to do all of those VERY computationally intensive tasks because we only care about our model's predictions.

I use the **DataLoader** in a similar way as in train to get out batches to feed to our model.

In the process I keep track of the actual labels and the predicted labels along with the loss.


```python
def validation(dataloader, device_):
  r"""Validation function to evaluate model performance on a 
  separate set of data.

  This function will return the true and predicted labels so we can use later
  to evaluate the model's performance.

  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

    dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

    device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

  Returns:
    
    :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss]
  """

  # Use global variable for model.
  global model

  # Tracking variables
  predictions_labels = []
  true_labels = []
  #total loss for this epoch.
  total_loss = 0

  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
  model.eval()

  # Evaluate data for one epoch
  for batch in tqdm(dataloader, total=len(dataloader)):

    # add original labels
    true_labels += batch['labels'].numpy().flatten().tolist()

    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up validation
    with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
        loss, logits = outputs[:2]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()
        
        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()

        # update list
        predictions_labels += predict_content

  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)

  # Return all true labels and prediciton for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss
```

<br>

## **Load Model and Tokenizer**

Loading the three essential parts of the pretrained transformers: *configuration*, *tokenizer* and *model*. I also need to load the
model on the device I'm planning to use (GPU / CPU).

Since I use the **AutoClass** functionality from **Hugging Face** I only need to worry about the model's name as input and the 
rest is handled by the transformers library.


```python
# Get model configuration.
print('Loading configuraiton...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path, 
                                          num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

# Get the actual model.
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, 
                                                           config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)
```

    Loading configuraiton...
    Loading tokenizer...
    Loading model...
    Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Model loaded to `cuda`


<br>

## **Dataset and DataLoader**

This is wehere I create the PyTorch **Dataset** and **DataLoader** objects that will be used to feed data into our model. 

This is where I use the **MovieReviewsDataset** class and create the dataset variables. Since data is partitioned 
for both train and test I will create a PyTorch **Dataset** and PyTorch **DataLoader** object for train and test. **ONLY for simplicity I will use 
the test as validation. In practice NEVER USE THE TEST DATA FOR VALIDATION!**


```python
print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = MovieReviewsDataset(path='/content/aclImdb/train', 
                               use_tokenizer=tokenizer, 
                               labels_ids=labels_ids,
                               max_sequence_len=max_length)
print('Created `train_dataset` with %d examples!'%len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with ...')
# Create pytorch dataset.
valid_dataset =  MovieReviewsDataset(path='/content/aclImdb/test', 
                               use_tokenizer=tokenizer, 
                               labels_ids=labels_ids,
                               max_sequence_len=max_length)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
```

    Dealing with Train...
    Reading partitions...
    100%|████████████████████████████████|2/2 [00:34<00:00, 17.28s/it]
    Reading neg files...
    100%|████████████████████████████████|12500/12500 [00:34<00:00, 362.01it/s]
    
    Reading pos files...
    100%|████████████████████████████████|12500/12500 [00:23<00:00, 534.34it/s]
    
    
    Using tokenizer on all texts. This can take a while...
    Texts padded or truncated to 40 length!
    Finished!
    
    Created `train_dataset` with 25000 examples!
    Created `train_dataloader` with 25000 batches!
    
    Dealing with ...
    Reading partitions...
    100%|████████████████████████████████|2/2 [01:28<00:00, 44.13s/it]
    Reading neg files...
    100%|████████████████████████████████|12500/12500 [01:28<00:00, 141.71it/s]
    
    Reading pos files...
    100%|████████████████████████████████|12500/12500 [01:17<00:00, 161.60it/s]
    
    
    Using tokenizer on all texts. This can take a while...
    Texts padded or truncated to 40 length!
    Finished!
    
    Created `valid_dataset` with 25000 examples!
    Created `eval_dataloader` with 25000 batches!


<br>

## **Train**

I create an optimizer and scheduler that will be used by PyTorch in training.

I loop through the number of defined epochs and call the **train** and **validation** functions.

I will output similar info after each epoch as in Keras: *train_loss:  - val_loss:  - train_acc: - valid_acc*.

After training, I plot the train and validation loss and accuracy curves to check how the training went.

```python
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives 
# us the number of batches.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}

# Loop through each epoch.
print('Epoch')
for epoch in tqdm(range(epochs)):
  print()
  print('Training on batches...')
  # Perform one full pass over the training set.
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)

  # Get prediction form model on validation data. 
  print('Validation on batches...')
  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)

  # Print loss and accuracy values to see how training evolves.
  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()

  # Store the loss value for plotting the learning curve.
  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)

# Plot loss curves.
plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])

# Plot accuracy curves.
plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
```

    Epoch
    100%|████████████████████████████████|4/4 [13:49<00:00, 207.37s/it]
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:40<00:00, 4.86it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [00:46<00:00, 16.80it/s]
    
      train_loss: 0.44816 - val_loss: 0.38655 - train_acc: 0.78372 - valid_acc: 0.81892
    
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:40<00:00, 4.86it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [02:13<00:00, 5.88it/s]
    
      train_loss: 0.29504 - val_loss: 0.43493 - train_acc: 0.87352 - valid_acc: 0.82360
    
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:40<00:00, 4.87it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [01:43<00:00, 7.58it/s]
    
      train_loss: 0.16901 - val_loss: 0.48433 - train_acc: 0.93544 - valid_acc: 0.82624
    
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:40<00:00, 4.87it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [00:46<00:00, 16.79it/s]
    
      train_loss: 0.09816 - val_loss: 0.73001 - train_acc: 0.96936 - valid_acc: 0.82144


![](data:image/png;base64,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3XffjWVZ3HHHHVRVVVFTU0NpaSm1tbWsXbuWlJQUpk2bRkVFBR9++CFlZWVs376d5cuXU1hYCMC0adNs73fPPfeQnp5Oeno6s2fP5tSpU8ydO5ekpCTOnj3LuXPn8Hg83HTTTdH8bRARiQqFXhGRGFm9evWQPb07d+7E7XZjWVb4tZycHPx+P+fOncPlcjFx4sTwNa/XS319PQA+n4+8vLyrvl9WVlb469TUVC5evAiEwvZbb73FE088QXp6OkuXLuV73/vedfmMIiLxQqFXRCTO+P1+jDHh4NvW1kZ5eTnZ2dlcuHCBnp6ecPBta2vD7XYD4PF4aGlpYerUqV/p/bKysvjVr34FQG1tLevXr6e0tJT8/Pzr+KlERGJL5/SKiMSZ8+fP8+677xIIBNi9ezdnzpzh1ltvxev1MmvWLN588016e3tpaGhgx44d3HnnnQBUVFSwbds2mpqaMMbQ0NBAZ2fniO+3e/dufD4fAOnp6QC2lWYRESfQSq+ISIy8+OKLtnN658yZw2233UZJSQlNTU2sWLGCrKwsHnvsMTIyMgB49NFHee211/jlL3+Jy+XivvvuC2+RWLp0KX19fTz77LN0dnZSVFTE448/PuI46uvreeONN+ju7iYrK4uf//zn/3GbhIjIWGQZY0ysByEiIiGXjyxbv359rIciIuIo2t4gIiIiIo6n0CsiIiIijqftDSIiIiLieFrpFRERERHHU+gVEREREcdT6BURERERx1PoFRERERHHU+gVEREREcdT6BURERERx/t/OGri/O9crHAAAAAASUVORK5CYII=)

![](data:image/png;base64,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xazNZoPFYnE/tlgssNls59339OnTKC0tRUpKSqtteXl5GDVqlMdzr7/+Oh566CHk5OTAbrdf3sCJiIiIyK/53XK2eXl5GD58OBTFs86uqKjA999/79HFYPr06YiMjITD4UB2djbee+893HTTTa2OuXHjRmzcuBEA8OSTT8JqtXr3Q5xDr9f79P2oBXOvDeZdO8y9dph7bTDv2vGn3PukmI2OjkZ5ebn7cXl5OaKjo8+779atW5GZmdnq+W3btmHo0KHQ61tCjoqKAgAYDAaMGzcO69evP+8xMzIykJGR4X5sNBp/1uf4uXz9ftSCudcG864d5l47zL02mHft+EvufdLNIDExEcXFxSgtLYXD4cDWrVuRlpbWar8TJ06gpqYGycnJrbadr4tBRUUFAEBKiR07diA+Pt47H+ASLFy4UOsQOi3mXhvMu3aYe+0w99pg3rXjT7n3ScusTqfDrFmzsHTpUqiqinHjxiE+Ph65ublITEx0F7Z5eXkYOXIkhBAery8tLUVZWRn69evn8fxzzz2Hs2fPAgB69eqF3/72t774OERERETkJ3zWZzY1NRWpqakez918880ej6dNm3be18bExCA7O7vV84sXL758ARIRERFRwNE99thjj2kdREd3xRVXaB1Cp8Xca4N51w5zrx3mXhvMu3b8JfdCSim1DoKIiIiI6OfwyQAwIiIiIiJv8Lt5ZgPRqlWrsGvXLkRERGD58uWttkspsXbtWnz99dcICgrC7Nmz/aZpPtBdLPd79+7FX/7yF8TExAAAhg0bdt65iOmnKSsrQ1ZWFiorKyGEQEZGBiZOnOixD89772hP7nnee0djYyMWL14Mh8MBp9OJ4cOHtxrrYbfbsXLlShw+fBhhYWGYO3eu+/+Bfp725H3z5s149dVX3dN+TpgwAenp6VqE2yGpqoqFCxciOjq61SwGfnHOS7pke/fulUVFRXLevHnn3b5z5065dOlSqaqqPHDggFy0aJGPI+y4Lpb7goIC+cQTT/g4qo7PZrPJoqIiKaWUtbW18v7775c//PCDxz48772jPbnnee8dqqrKuro6KaWUdrtdLlq0SB44cMBjnw0bNsjs7GwppZRffvmlfOaZZ3weZ0fTnrxv2rRJ/u1vf9MivE5h/fr1csWKFee9rvjDOc9uBpdBv379YDab29yen5+PMWPGQAiB5ORk1NTU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)

It looks like a little over one epoch is enough training for this model and dataset.

<br>

## **Evaluate**

When dealing with classification it's useful to look at precision recall and f1 score. 
Another good thing to look at when evaluating the model is the confusion matrix.


```python
# Get prediction form model on validation data. This is where you should use
# your test data.
true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader, device)

# Create the evaluation report.
evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
# Show the evaluation report.
print(evaluation_report)

# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels, 
                      classes=list(labels_ids.keys()), normalize=True, 
                      magnify=3,
                      );
```

    100%|████████████████████████████████|782/782 [00:46<00:00, 16.77it/s]
    
                  precision    recall  f1-score   support
    
             neg       0.83      0.81      0.82     12500
             pos       0.81      0.83      0.82     12500
    
        accuracy                           0.82     25000
       macro avg       0.82      0.82      0.82     25000
    weighted avg       0.82      0.82      0.82     25000
    
    Normalized confusion matrix

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA44AAAKyCAYAAACaOID2AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeZyVdd0//teZYRdcGGURSHNcETVt3JAUE9FCjcotf2ak9bD8mtymJZbmUiiaelda1p3caNZdaJaVBC7lBpSChFsu4JbIEAoumaAMc35/wD23E3IYT8wi5/nscR6Pc51znet6n5mpfPt6X5+rUCwWiwEAAIC1qGrvAgAAAOjYNI4AAACUpHEEAACgJI0jAAAAJWkcAQAAKEnjCAAAQEkaR+A9qVAo5Kc//WnT9tZbb51vfetbrXrOu+66K4VCIQsWLGjV87TEa6+9lo9//OPZZJNNUigU8uyzz66X4/7rz3VD15F+pwDQkWkcoUKNGTMmhUIhX/3qV5u9vmDBghQKhdx1113tU1iZZs2aldNPP729y2gzV199df70pz9l+vTpqa+vz6BBg9bLcevr63PkkUeul2O1l2233Tbnn39+i/YdOnRo6uvrs+WWW7ZuUQDwHqdxhArWrVu3fO9738tzzz23Xo9bLBazYsWK9XrMddliiy2y0UYbtek529O8efOy8847Z5dddkm/fv1SXV29Xo7br1+/dOvWbb0cq6NbsWJFunTpkn79+qWqyv8dAkAp/p8SKtjQoUOz22675Wtf+1rJ/Z544omMGjUqPXv2TM+ePXP44Ydn/vz5Te9fe+216dSpU+68887svvvu6dq1a+64444MHz48J510Us4555z06dMnm266ab7+9a+nsbExF154Yfr27ZstttgiX//615ud73/+53+y9957Z5NNNsnmm2+eUaNG5cknnyxZ49tHVa+99toUCoU1HsOHD2/a/4EHHsjIkSPTs2fPbLHFFvnEJz6xRgN95ZVXZuDAgenRo0cOOeSQ/O1vf2vJjzXf//73M3jw4HTt2jV9+vTJJz/5yab3/vGPf+Tkk0/OFltska5du6auri633XZb0/vPPvtsCoVCbrjhhhx22GHp0aNHttlmm1x77bXNvuvEiRPzxz/+sdn3eqdx3c997nPNvvf06dOz3377pVevXunVq1d222233HrrrU3v/+uoan19fY499thsuumm6d69e4YPH57Zs2c3vf+/o56333579t9///To0SODBw/O1KlTS/6Mzj///Gy77ba54YYbst1226VHjx4ZPXp0XnvttfzqV7/KDjvskF69euXII4/Mq6++2vS5OXPm5CMf+Uj69OmTnj17Zs8998y0adOa3h8+fHieeuqpXHDBBU2/92effbapzilTpmTYsGHp1q1brrnmmjVGVS+99NJsuummzUZ/L7zwwmyxxRZZuHBhye8EABsyjSNUsEKhkMsuuyw///nPmzUDb7ds2bKMHDkyy5cvz91335277747r7/+eg499NC89dZbTfs1NjbmrLPOyhVXXJHHH388dXV1SZJf/vKXWbFiRaZPn54rrrgiF110UUaNGpXXX3899957by677LJcdNFFzRqNN998M+ecc07mzJmT22+/PdXV1Rk1alSz85VyzDHHpL6+vukxc+bM9OrVKwceeGCS5K9//WsOOOCA7Lvvvpk9e3b++Mc/prq6OgcffHCWL1+eJPnNb36T008/PV/+8pczd+7cHH300fnKV76yznOfd955Oeuss3LKKafk4YcfzrRp07LHHns0vX/iiSfm1ltvzU9/+tPMnTs3++23Xw477LA8/vjjzY4zbty4nHDCCXnooYdy7LHH5nOf+1xT8zxr1qwcffTR+dCHPpT6+vr86le/atHPpaGhIUcccUT23nvvzJkzJ3PmzMn555+fHj16vOP+xWIxo0ePzuOPP55bbrkl999/f/r27ZuDDz44L730UrN9zzzzzHzta1/Lgw8+mL333jvHHHNMXn755ZL11NfX57rrrstNN92UqVOnZsaMGTnyyCNzzTXX5IYbbsjUqVNz77335qKLLmr6zGuvvZZjjjkmd955Z+bMmZNDDjkkRxxxRNPP5le/+lW23nrrnHHGGU2//7eP8Z5xxhk566yz8thjj+Xwww9fo6avfOUr2XvvvfOpT30qDQ0Nueeee/LNb34z1157rXFWACpbEahIn/nMZ4oHHXRQsVgsFkePHl084IADisVisfj8888XkxTvvPPOYrFYLF5zzTXF7t27F1988cWmzy5atKjYrVu34nXXXVcsFovFSZMmFZMU77nnnmbnOOCAA4q77bZbs9cGDx5cHDJkSLPXdt111+IZZ5yx1lqXLFlSTFKcPn1602tJitdff33T9lZbbVX85je/ucZnX3nlleLgwYOLRx99dLGxsbHpux9zzDHN9lu+fHmxe/fuxV//+tfFYrFY3G+//YrHHXdcs33OOOOMYpLi888//451vv7668Vu3boVv/3tb7/j+/PmzSsmKU6ZMqXZ67vvvnvxs5/9bLFYLBafeeaZYpLi5Zdf3vR+Q0NDsWfPnsUf/vCHTa+9/fdX6mdw0kknNf1uly5d2ux3+07e/nO94447ikmKjz76aNP7y5cvL/br1694wQUXFIvFYvHOO+8sJinedNNNTfssWrSomKQ4bdq0tZ7nvPPOK1ZXVzf7uzrllFOKVVVVxcWLFze9dtpppxU/+MEPrvU4xeKqv59vfetbTdu1tbXF8847r9k+/1vnT37yk3d8/e2/00WLFhX79u1b/OIXv1gcOHBg8T/+4z9Knh8AKoHEEcgll1ySGTNm5Le//e0a7z366KMZPHhwNt9886bX+vbtmx122CGPPvpos3333HPPNT6/2267Ndvu169fdt111zVeW7x4cdP23Llz8/GPfzzvf//706tXr7zvfe9Lknd9LWZDQ0OOPvrobLzxxrnuuutSKBSSrErsfv3rXzeN3vbs2TM1NTVZvnx55s2bl2RVKjl06NBmxxs2bFjJ8z366KNZvnx5Ro4c+Y7v//Wvf02S7L///s1e33///df4WX7gAx9oel5dXZ0+ffrk73//ewu+9dptttlm+dznPpdDDjkkH/nIRzJhwoQ88cQTa93/0UcfTU1NTQYPHtz0WteuXbP33nuXrLdv376prq5eZ70DBgxo9nfVr1+/9OvXL1tssUWz197+t/Hiiy/mlFNOyY477phNN900PXv2zKOPPtriv4299tprnfv07ds3kyZNytVXX52amppccsklLTo2AGzINI5Att9++5x88sk566yz0tDQUNYxqqur33FRlc6dOzfbLhQK7/haY2NjkuSNN97IyJEjUygUMmnSpNx///2ZNWtWCoVCi0dV/9fYsWPz5JNP5je/+U2z2hobG/PpT386c+fObfZ48skn87nPfe5dnaO1dOnSpdn2239Ga1NVVZVisdjstX9dpOjHP/5xHnjggRx88MG5++67M2TIkPzoRz9a7/UmWWe97/ZvI1m1GvC9996bSy+9NPfee2/mzp2bD3zgAy3+22jpAkp33313U/P79mssAaBSaRyBJKuuzVu4cGH+67/+q9nrO++8c/761782u6bt73//e5544okMGTJkvdfx2GOP5cUXX8z48eMzfPjw7LTTTnn55ZfXaIjW5Xvf+15+9rOfZcqUKenTp0+z9+rq6vLQQw+ltrY22267bbPHZpttliQZPHhwZs6c2exzM2bMKHnOwYMHp1u3bs0Wu3m7nXfeOUlyzz33NHv9nnvuWS8/yz59+qyxgMtf/vKXNfYbMmRIvvzlL2fq1Kk56aST1vidv73eJUuWNCWlyarrT++7775W+d23xD333JNTTjklRxxxRHbZZZf0798/Tz/9dLN9unTpkpUrV5Z9jjvuuCOXX355brnllgwaNChjxox5139/ALCh0TgCSVbdzmLcuHH5zne+0+z14447LltssUWOOeaYzJkzJw888ECOPfbYDBgwIMccc8x6r2OrrbZK165dc+WVV+app57KH/7wh4wdO7ZpzLQl7rjjjpxxxhm58sor07t37yxatCiLFi3K0qVLkyRf+9rX8thjj+X444/P/fffn2eeeSZ33nlnxo4d29SEnHHGGZk8eXK++93vZt68eZk0aVKuv/76kuft2bNnzjjjjJx//vn5/ve/nyeffDIPPvhgLr744iRJbW1tjjrqqJxyyim59dZb8/jjj2fs2LF55JFHWrTwzrqMGDEikydPzm233ZYnnngip59+erMRzvnz5+ess87K9OnT89xzz+VPf/pT7r333majqG/34Q9/OHvttVeOO+64zJgxI4888khOOOGELF++PF/84hf/7XrLscMOO+RnP/tZHn744cydOzef+tSn1mgS3//+92fGjBn529/+lpdeemmdyefbvfjii/n0pz+dr3zlKzn00EPz85//PPfee+8a/70AgEqjcQSanH766c2uOUuS7t2757bbbkvXrl2z//7754ADDshGG22UadOmveN44r9r8803z09/+tPcfvvt2XnnnXPmmWfmsssue1f32Zs+fXoaGhpywgknpH///k2PT3ziE0mSnXbaKTNnzszrr7+eQw45JIMHD87nP//5LFu2LJtuummS5OMf/3guv/zyXHrppdl1113zs5/9rEXXun3zm9/M+PHj873vfS9DhgzJyJEjM2fOnKb3r7nmmhxyyCE5/vjjs9tuu2XGjBm55ZZbsuOOO77Ln9SazjrrrIwaNSrH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AKAMVZUTOEocAQAAKE3jCAAAQElGVQEAAMpgcRwAAABYTeIIAABQhkLlBI4SRwAAAErTOEKS226dll133iE777htvn3phDXen37vPdl3zz3Ss1un/OqmXza9fvddd2bvD36g6bFpz2757W9ubsvSAZo5eOhOefDX5+aR35yXMz978BrvD+q3Wab912n508/Pyv2Tz84hwwYnSep23ip//sW4/PkX43Lf5HE54sBd27p0gPecQjs82kuhWCwW2/H869XyhvaugPeilStXZpfB22fK1NszYODADNtnz1z3059np8GDm/Z57tln89prr+U7V1yWUYcfkU988sg1jrN06dIM2XHbzH92QXr06NGWX4ENyGZ7ntreJfAeVlVVyMM3fyOjvnhVXvj7K5n+s6/kM2dfm8efXtS0z1XnfCoPPvF8fnzj9Oy4Tb/cfOUXs+Oo89K9W+e8tWJlVq5sTL/NN859k8/ONiO/npUrG9vxG/FetuwvV7V3CdDqZsx7uc3Pud92m7X5OROJI2TW/fentnbbvH+bbdKlS5ccdcyxueV3v2m2z1Zbb51ddt01VVVr/6/Mr2/6ZUYe8hFNI9Bu9hyydZ56/qU8+8KSrGhYmRtvnZPDhjdPDovFYjbeqFuSZJOe3VP/4qtJkmXLVzQ1iV27dM4G9O+VAVgP2mxxnMWLF+eiiy7KNttsk2eeeSYDBw7MqaeemieffDLXX399Vq5cmdra2nz+859P586d87Of/SyzZ89OdXV1dt1115xwwgltVSoVZuHCFzJw4KCm7QEDBub+++9718e58YZf5LT/+PL6LA3gXdmyzyZZ8Pf/+7ffL/z95ew1ZOtm+4z/0e/zux+cmi8ee0B6dO+aUV+4sum9PYdslR+ef3ze1793TjrnOmkjwDpUVdDqOG26qurChQvzhS98ITvuuGN+8IMf5JZbbskdd9yRc889N1tuuWWuuuqq3Hbbbdl///1z//335zvf+U4KhUL++c9/tmWZ8K7V19fn0UcezsEjD2nvUgBKOvrQuvz0d3/Od6//Y/be9f2Z+K0T8sEjL0qxWMysR57LB48cnx3e3zfXXPjp3Drjr3nzLdeBANDGo6o1NTXZcccdkyT7779/HnnkkfTp0ydbbrllkuSAAw7IY489lh49eqRLly65+uqrc99996Vr167veLw77rgj48aNy7hx49rs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KEvlRI4SRwAAAAopHAEAACikVRUAAKAMNscBAACAf5E4AgAAlKGCAkeJIwAAAMUUjgAAABTSqgoAAFAGm+MAAADAv0gcAQAAylCqoO1xJI4AAAAUkjgCAACUo3ICR4kjAAAAxRSOAAAAFNKqCgAAUIYK6lSVOAIAAFBM4ggAAFCGUgVFjhJHAAAACikcAQAAKKRVFQAAoAylCtoeR+IIAABAIYkjAABAOSoncJQ4AgAAUEziCAAAUIYKChwljgAAABRTOAIAAFBIqyoAAEAZShXUqypxBAAAoJDEEQAAoAylg3R7nHXr1uWOO+5IZ2dnJkyYkPPPP3+v4ytXrsyqVatSXV2d/v3754orrsigQYMK55Q4AgAAvE90dnZm0aJF+frXv57Zs2fn4YcfztatW/cac9xxx+Xmm2/O97///Zx55pm566679juvwhEAAOB9YvPmzRk8eHAaGhpSU1OTT3ziE3nsscf2GnPKKaekd+/eSZITTzwxbW1t+51X4QgAAFCGUum9/7c/bW1tqaur635dV1dXWBj+9re/zciRI/c7r2ccAQAADiEzZszo/nrixImZOHFiWfOsXr06W7ZsyU033bTfsQpHAACAQ8jNN9/8b48NHDgwra2t3a9bW1szcODAt41bv359fvKTn+Smm27KYYcdtt9zalUFAAB4nxg2bFiam5vz8ssvp729PY888khGjRq115gXXnghCxcuzHXXXZcjjzzygOaVOAIAAJThQJ45fK9VV1fn0ksvzXe+8510dnbmU5/6VIYMGZLly5dn2LBhGTVqVO66667s3r07t956a5Kkvr4+119/feG8pa6urq734gLeC7vbe3oFAO/OUaP/p6eXAPAfseuJH/T0EuC/bvuujvf8nAP6Vr/n50y0qgIAALAfWlUBAADKUMpB2Kv6XyJxBAAAoJDEEQAAoAwH4+Y4/y0SRwAAAAopHAEAACikVRUAAKAMFdSpKnEEAACgmMQRAACgHBUUOUocAQAAKCRxBAAAKEOpgiJHiSMAAACFFI4AAAAU0qoKAABQhlLldKpKHAEAACgmcQQAAChDBQWOEkcAAACKKRwBAAAopFUVAACgHBXUqypxBAAAoJDEEQAAoAylCoocJY4AAAAUkjgCAACUoVQ5gaPEEQAAgGKlrq6urp5eBAAAAAcviSO8AzNmzOjpJQC8az7LAHinFI4AAAAUUjgCAABQSOEI78DEiRN7egkA75rPMgDeKZvjAAAAUEjiCAAAQCGFIwAAAIUUjgAAABRSOAIAAFBI4QgH6M19pOwnBbxf+DwD4EApHOEANTc3J0lKpZL/bAGHtKampmzfvj2lUqmnlwLAIULhCAegubk5X/va17Jo0aIkikfg0LV27dr88Ic/zLZt27rf83kGwP7U9PQC4GC3du3a/P73v89nPvOZrF69Oh0dHbnsssu6i0e/sQcOFU1NTVm2bFmmT5+ewYMH59VXX83rr7+e+vr6dHZ2pqrK75MB2Dc/IaDA7t27s3LlyowbNy4XXXRRbrnlljzzzDNZvHhxEskjcGh483Nqx44dOfLII7Njx46sWLEic+fOzfTp0/PXv/5V0QhAIT8loEDv3r1z9NFHp66uLklSW1ubL3/5y3nooYeybNmyJJE4Age9nTt3JklGjBiR448/PkuWLMnRRx+da665Juedd16ampp6eIUAHOy0qsI+vPjii+nVq1dqa2tzwgkn5Pbbb8/MmTPTu3fv9OnTJxMnTsz69evT2NiYESNG9PRyAf6tdevWZeXKlRkwYEAGDRqU888/PxdddFGSZNOmTVm9enWuuOKKHl4lAAc7hSP8P0888USWLl2aM888Mw8//HBuueWWNDU15cYbb8xHP/rRrFmzJtddd12qqqq0dgEHtaampixatChXXnlldu3alS1btmThwoWZOnVqdu7cmblz5+ZLX/pSPvzhD/f0UgE4yCkc4S1eeumlrFixItOnT8/mzZtTKpWyZ8+eTJs2LU8//XT27NmTT3/609mxY0eefPLJnHPOOT29ZIB/64033khjY2NOPvnkdHZ2ZujQobnnnnvy4osv5iMf+UiuvfbaHHPMMTb6AmC/FI7wFocffnjGjRuXLVu25IEHHsh1112Xvn375sknn8yJJ56Yfv36pampKXfddVeuuuqqNDQ09PSSAd7m2Wefzcsvv5yOjo788Y9/zOmnn57TTjstdXV1qa6uzrZt21JVVZVjjjkmiWe1Adg/hSMk2bBhQ7Zu3ZqGhoY88MAD6ejoyJw5c1JTU5NNmzblpz/9aS6//PL069cvdXV1mTFjRo444oieXjbA2zz33HNZsGBBPvShD2XAgAGpr6/Pvffem9bW1gwZMiSbNm3KJz/5yZ5eJgCHmFKXvyVAhXv++eczb968fOADH8gHP/jBvP7661m9enUmT56c6urqPPjgg7ngggsyevTonl4qQKHNmzdn6dKl+cIXvpDhw4fn73//e9auXZvnnnsur732Wurr63P66afnjDPO6OmlAnCIkThS0TZv3pwf//jHufrqqzN06NCsXr0627Zty5gxY7J169YMGTIkX/ziF9PY2OgZIOCg989//jMbNmzI008/neHDh6euri4NDQ1pbW3NNddc072hl88zAN4pW0JS0V577bU89dRTWb9+fZJk7NixaWhoSN++fXPsscdm0qRJaWxsTOIZIODg19jYmOnTp+fBBx/MmjVrUlNTk379+mXjxo3ZuXNn3mwy8nkGwDulVZWKt3bt2vzoRz/KlClTMm7cuHR2duaRRx7J0KFDM2TIkJ5eHsA7tnbt2syZMyeNjY0plUo5++yzM2rUqJ5eFgCHMIUjJHn88cezzse2TwAABqZJREFUfPnyfPazn8348eN7ejkA79ratWuzfPnynHXWWTnvvPOkjQC8K55xhCSnnXZaOjs7s3Tp0jQ2NmbAgAHdzwIBHIpGjRqVww47LPPmzUtDQ0M+/vGP9/SSADiESRzhLV599dX079+/p5cB8B+zfv36NDQ0+LuzALwrCkcAAAAK6cUDAACgkMIRAACAQgpHAAAACikcAQAAKKRwBHgX5s6dm2XLliVJNm7cmKuvvvo9Oe/nP//5vPTSS/s8dtNNN2XVqlUHNM9VV12V9evXl7WGd/O9AMChxd9xBN73rrrqqmzfvj1VVVXp06dPRo4cmWnTpqVPnz7/0fOcfPLJue222/Y77qGHHsqqVavyrW996z96fgCA/xaJI1ARrr/++tx5552ZOXNmtmzZknvvvfdtYzo6OnpgZQAABz+JI1BRBg4cmJEjR6apqSnJ/7V8Xnrppfn5z3+ejo6OzJ07N3/+85+zbNmybNu2Lcccc0y++tWvZujQoUmSF154IfPnz09zc3NOPfXUlEql7rmfeeaZzJkzJ/Pnz0+StLS0ZMmSJdm4cWO6uroyduzYnHvuuVm4cGHa29szderUVFdXZ8mSJXnjjTdy99135w9/+EPa29szevToXHLJJenVq1eS5Gc/+1lWrlyZUqmUCy+88ICv96WXXsqCBQvyt7/9LaVSKR/72Mcybdq0HH744d1j/vKXv+SOO+7I9u3bM3r06HzlK1/pPm/RvQAAKofEEagoLS0teeKJJ3Lcccd1v/fYY4/lu9/9bmbPnp0XXngh8+bNy2WXXZbFixdn4sSJ+d73vpc33ngj7e3tmTVrVs4666wsXrw4Y8aMyaOPPrrP83R2dmbmzJmpr6/P3LlzM3/+/IwdO7a7+Bo+fHjuvPPOLFmyJEmydOnSNDc3Z9asWbn99tvT1taWFStWJEnWrVuX+++/P9/4xjdy22235amnnnpH1zx58uQsWLAgs2fPTmtra+655569jq9ZsyY33HBD5syZk+bm5tx3331JUngvAIDKonAEKsKsWbNyySWX5MYbb8yIESMyZcqU7mOTJ09ObW1tevXqld/85jeZOHFiTjzxxFRVVWX8+PGpqanJ888/n02bNqWjoyOTJk1KTU1NzjzzzAwbNmyf59u8eXPa2toyderU9OnTJ7169cpJJ520z7FdXV1ZtWpVLr744tTW1qZv376ZMmVKHn744STJI488kvHjx+fYY49Nnz59csEFFxzwdQ8ePDiNjY057LDD0r9//0yaNCkbNmzYa8y5556b+vr61NbWZvLkyd3nLboXAEBl0aoKVIRrr702jY2N+zxWV1fX/XVLS0t+97vf5Ze//GX3e+3t7Wlra0upVMrAgQP3ak+tr6/f55wtLS0ZNGhQqqur97u2V199NXv27MmMGTO63+vq6kpnZ2eS5JVXXsnxxx/ffWzQoEH7nfNN27dv726X3b17dzo7O1NbW7vXmLdew6BBg9LW1tZ9Df/uXgAAlUXhCFS8txaCdXV1mTJlyl6J5Js2bNiQtra2dHV1dX9Pa2trBg8e/Lax9fX1aWlpSUdHx36LxyOOOCK9evXKrbfemoEDB77t+FFHHZXW1tbu1y0tLQd8bXfffXeS5JZbbkltbW3+9Kc/ZfHixXuNeet8LS0t3WsouhcAQGXRqgrwFhMmTMivf/3rPP/88+nq6sru3bvz+OOPZ9euXRk+fHiqqqryi1/8Iu3t7Xn00UezefPmfc5zwgkn5KijjsrSpUuze/fuvP7663n22WeTJAMGDEhbW1va29uTJFVVVZkwYUKWLFmSHTt2JEna2tqybt26JMmYMWPy0EMPZevWrdmzZ8/bnlEssmvXrvTp0yf9+vVLW1tb7r///reN+dWvfpXW1tb84x//yH333ZcxY8bs914AAJVF4gjwFsOGDcvll1+exYsXp7m5ufvZxJNPPjk1NTWZPn16FixYkGXLluXUU0/NGWecsc95qqqqcv3112fx4sW58sorUyqVMnbs2Jx00kk55ZRTujfJqaqqyqJFi3LRRRdlxYoVueGGG7Jz584MHDgw55xzTkaOHJlTTz01kyZNyje/+c1UVVXlwgsvzJo1aw7oei644IL84Ac/yMUXX5zBgwfn7LPPzgMPPLDXmHHjxuXb3/52XnnllYwaNSqf+9zn9nsvAIDKUurq6urq6UUAAABw8NKqCgAAQCGFIwAAAIUUjgAAABRSOAIAAFBI4QgAAEAhhSMAAACFFI4AAAAUUjgCAABQ6H8BHVMQ4UOwTMgAAAAASUVORK5CYII=)

Results are not great, but for this tutorial we are not interested in performance.

<br>

## **Final Note**

If you made it this far Congrats :confetti_ball: and Thank you :pray: for your interest in my tutorial!

I've been using this code for a while now and I feel it got to a point where is nicely documented and easy to follow.

Of course is easy for me to follow because I built it. That is why any feedback is welcome and it helps me improve my future tutorials!

If you have 1 minute please give me a feedback in the comments.

If you see something wrong please let me know by opening an 
**[issue on my ml_things](https://github.com/gmihaila/ml_things/issues/new/choose)** GitHub repository! 

A lot of tutorials out there are mostly a one-time thing and are not being maintained. I plan on keeping my 
tutorials up to date as much as I can.

<br>

## **Contact** 🎣

🦊 GitHub: [gmihaila](https://github.com/gmihaila)

🌐 Website: [gmihaila.github.io](https://gmihaila.github.io/)

👔 LinkedIn: [mihailageorge](https://medium.com/r/?url=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fmihailageorge)

📬 Email: [georgemihaila@my.unt.edu.com](mailto:georgemihaila@my.unt.edu.com?subject=GitHub%20Website)

<br>