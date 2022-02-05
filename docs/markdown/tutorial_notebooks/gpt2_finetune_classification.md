# 🎱 **GPT2 For Text Classification using Hugging Face 🤗 Transformers**

## **Complete tutorial on how to use GPT2 for text classification.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb) &nbsp;
[![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb)
[![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/6t6kvlewoabwxqw/gpt2_finetune_classification.ipynb?dl=1)
[![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://gmihaila.medium.com/gpt2-for-text-classification-using-hugging-face-transformers-574555451832)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<br>

**Disclaimer:** *The format of this tutorial notebook is very similar to my other tutorial notebooks. This is done intentionally in order to keep readers familiar with my format.*

<br>

This notebook is used to fine-tune GPT2 model for text classification using [Huggingface](https://huggingface.co/transformers/) [transformers](https://github.com/huggingface/transformers) library on a custom dataset.

Hugging Face is very nice to us to include all the functionality needed for GPT2 to be used in classification tasks. Thank you Hugging Face! 

I wasn't able to find much information on how to use GPT2 for classification so I decided to make this tutorial using similar structure with other transformers models.

**Main idea:**
Since GPT2 is a decoder transformer, the last token of the input sequence is used to make predictions about the next token that should follow the input. This means that the last token of the input sequence contains all the information needed in the prediction. With this in mind we can use that information to make a prediction in a classification task instead of generation task.

In other words, instead of using first token embedding to make prediction like we do in Bert, we will use the last token embedding to make prediction with GPT2.

Since we only cared about the first token in Bert, we were padding to the right. Now in GPT2 we are using the last token for prediction so we will need to pad on the left. Because of a nice upgrade to HuggingFace Transformers we are able to configure the GPT2 Tokenizer to do just that.


<br>

## **What should I know for this notebook?**

Since I am using PyTorch to fine-tune our transformers models any knowledge on PyTorch is very useful.

Knowing a little bit about the [transformers](https://github.com/huggingface/transformers) library helps too.

<br>

## **How to use this notebook?**


Like with every project, I built this notebook with reusability in mind.

All changes will happen in the data processing part where you need to customize the PyTorch Dataset, Data Collator and DataLoader to fit your own data needs.

All parameters that can be changed are under the **Imports** section. Each parameter is nicely commented and structured to be as intuitive as possible.

<br>

## **Dataset**

This notebook will cover pretraining transformers on a custom dataset. I will use the well known movies reviews positive - negative labeled [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

The description provided on the Stanford website:

*This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details.*

**Why this dataset?** I believe is an easy to understand and use dataset for classification. I think sentiment data is always fun to work with.

<br>

## **Coding**

Now let's do some coding! We will go through each coding cell in the notebook and describe what it does, what's the code, and when is relevant - show the output.

I made this format to be easy to follow if you decide to run each code cell in your own python notebook.

When I learn from a tutorial I always try to replicate the results. I believe it's easy to follow along if you have the code next to the explanations.

<br>
 

## **Downloads**

Download the *Large Movie Review Dataset* and unzip it locally.



```
# download the dataset
!wget -q -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# unzip it
!tar -zxf /content/aclImdb_v1.tar.gz
```

## **Installs**

* **[transformers](https://github.com/huggingface/transformers)** library needs to be installed to use all the awesome code from Hugging Face. To get the latest version I will install it straight from GitHub.

* **[ml_things](https://github.com/gmihaila/ml_things)** library used for various machine learning related tasks. I created this library to reduce the amount of code I need to write for each machine learning project.



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


## **Imports**

Import all needed libraries for this notebook.

Declare parameters used for this notebook:

* `set_seed(123)` - Always good to set a fixed seed for reproducibility.
* `epochs` - Number of training epochs (authors recommend between 2 and 4).
* `batch_size` - Number of batches - depending on the max sequence length and GPU memory. For 512 sequence length a batch of 10 USUALY works without cuda memory issues. For small sequence length can try batch of 32 or higher.
max_length - Pad or truncate text sequences to a specific length. I will set it to 60 to speed up training.
* `device` - Look for gpu to use. Will use cpu by default if no gpu found.
* `model_name_or_path` - Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk. In this tutorial I will use `gpt2` model.
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
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

# Set seed for reproducibility.
set_seed(123)

# Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
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
model_name_or_path = 'gpt2'

# Dictionary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'neg': 0, 'pos': 1}

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)
```

## **Helper Functions**

I like to keep all Classes and functions that will be used in this notebook under this section to help maintain a clean look of the notebook:

<br>

**MovieReviewsDataset(Dataset)**

If you worked with PyTorch before, this is pretty standard. We need this class to read in our dataset, parse it and return texts with their associated labels.

In this class I only need to read in the content of each file, use fix_text to fix any Unicode problems and keep track of positive and negative sentiments.

I will append all texts and labels in lists.

There are three main parts of this PyTorch Dataset class:

* **init()** where we read in the dataset and transform text and labels into numbers.
* **len()** where we need to return the number of examples we read in. This is used when calling len(MovieReviewsDataset()).
* **getitem()** always takes as an input an int value that represents which example from our examples to return from our dataset. If a value of 3 is passed, we will return the example form our dataset at position 3.

<br>

**Gpt2ClassificationCollator**

I use this class to create the Data Collator. This will be used in the DataLoader to create the bathes of data that get fed to the model. I use the tokenizer and label encoder on each sequence to convert texts and labels to number.

Lucky for us, Hugging Face thought of everything and made the tokenizer do all the heavy lifting (split text into tokens, padding, truncating, encode text into numbers) and is very easy to use!

There are two main parts of this Data Collator class:

* **init()** where we initialize the tokenizer we plan to use, how to encode our labels and if we need to set the sequence length to a different value.

* **__call__()** used as function collator that takes as input a batch of data examples. It needs to return an object with the format that can be fed to our model. Luckily our tokenizer does that for us and returns a dictionary of variables ready to be fed to the model in this way: `model(**inputs)`. Since we are fine-tuning the model I also included the labels.





<br>

**train(dataloader, optimizer_, scheduler_, device_)**

I created this function to perform a full pass through the DataLoader object (the DataLoader object is created from our Dataset* type object using the **MovieReviewsDataset class). This is basically one epoch train through the entire dataset.

The dataloader is created from PyTorch DataLoader which takes the object created from MovieReviewsDataset class and puts each example in batches. This way we can feed our model batches of data!

The optimizer_ and scheduler_ are very common in PyTorch. They are required to update the parameters of our model and update our learning rate during training. There is a lot more than that but I won't go into details. This can actually be a huge rabbit hole since A LOT happens behind these functions that we don't need to worry. Thank you PyTorch!

In the process we keep track of the actual labels and the predicted labels along with the loss.

<br>

**validation(dataloader, device_)**

I implemented this function in a very similar way as train but without the parameters update, backward pass and gradient decent part. We don't need to do all of those VERY computationally intensive tasks because we only care about our model's predictions.

I use the DataLoader in a similar way as in train to get out batches to feed to our model.

In the process I keep track of the actual labels and the predicted labels along with the loss.



```python
class MovieReviewsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind: it can be used as is as.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self, path, use_tokenizer):

    # Check if path exists.
    if not os.path.isdir(path):
      # Raise error if path is invalid.
      raise ValueError('Invalid `path` variable! Needs to be a directory')
    self.texts = []
    self.labels = []
    # Since the labels are defined by folders with data we loop 
    # through each label.
    for label in ['pos', 'neg']:
      sentiment_path = os.path.join(path, label)

      # Get all files from path.
      files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
      # Go through each file and read its content.
      for file_name in tqdm(files_names, desc=f'{label} files'):
        file_path = os.path.join(sentiment_path, file_name)

        # Read content.
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        # Fix any unicode issues.
        content = fix_text(content)
        # Save content.
        self.texts.append(content)
        # Save encode labels.
        self.labels.append(label)

    # Number of exmaples.
    self.n_examples = len(self.labels)
    

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
      :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
      asociated labels.

    """

    return {'text':self.texts[item],
            'label':self.labels[item]}



class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

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

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs


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

## **Load Model and Tokenizer**

Loading the three essential parts of the pretrained GPT2 transformer: configuration, tokenizer and model. 

For this example I will use `gpt2` from HuggingFace pretrained transformers. You can use any variations of GP2 you want.

In creating the `model_config` I will mention the number of labels I need for my classification task. Since I only predict two sentiments: positive and negative I will only need two labels for `num_labels`.

Creating the `tokenizer` is pretty standard when using the Transformers library. After creating the tokenizer it is critical for this tutorial to set padding to the left `tokenizer.padding_side = "left"` and initialize the padding token to `tokenizer.eos_token` which is the GPT2's original end of sequence token. This is the most essential part of this tutorial since GPT2 uses the last token for prediction so we need to pad to the left.

HuggingFace already did most of the work for us and added a classification layer to the GPT2 model. In creating the model I used `GPT2ForSequenceClassification`.
Since we have a custom padding token we need to initialize it for the model using `model.config.pad_token_id`.
Finally we will need to move the model to the device we defined earlier.




```python
# Get model configuration.
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token


# Get the actual model.
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)

# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))

# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)
```

    Loading configuraiton...
    Loading tokenizer...
    Loading model...
    Some weights of GPT2ForSequenceClassification were not initialized from the model checkpoint at gpt2 and are newly initialized: ['score.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    Model loaded to `cuda`

<br>

## **Dataset and Collator**

This is where I create the PyTorch Dataset and Data Loader with Data Collator objects that will be used to feed data into our model.

This is where I use the **MovieReviewsDataset** class to create the PyTorch Dataset that will return texts and labels.

Since we need to input numbers to our model we need to convert the texts and labels to numbers. This is the purpose of a collator! It takes data outputted by the PyTorch Dataset and passed through the Data Collator function to output the sequence for our model.

I'm keeping the tokenizer away from the PyTorch Dataset to make the code cleaner and better structured. You can obviously use the tokenizer inside the PyTorch Dataset and output sequences that can be used straight into the model without using a Data Collator.

I strongly recommend to use a validation text file in order to determine how much training is needed in order to avoid overfitting. After you figure out what parameters yield the best results, the validation file can be incorporated in train and run a final train with the whole dataset.

The data collator is used to format the PyTorch Dataset outputs to match the inputs needed for GPT2.


```python
# Create data collator to encode text and labels into numbers.
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)


print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = MovieReviewsDataset(path='/content/aclImdb/train', 
                               use_tokenizer=tokenizer)
print('Created `train_dataset` with %d examples!'%len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

print()

print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset =  MovieReviewsDataset(path='/content/aclImdb/test', 
                               use_tokenizer=tokenizer)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))
```

    Dealing with Train...
    pos files: 100%|████████████████████████████████|12500/12500 [01:17<00:00, 161.19it/s]
    neg files: 100%|████████████████████████████████|12500/12500 [01:05<00:00, 190.72it/s]
    
    Created `train_dataset` with 25000 examples!
    Created `train_dataloader` with 782 batches!
    Reading pos files...
    pos files: 100%|████████████████████████████████|12500/12500 [00:54<00:00, 230.93it/s]
    neg files: 100%|████████████████████████████████|12500/12500 [00:42<00:00, 291.07it/s]
    
    Created `valid_dataset` with 25000 examples!
    Created `eval_dataloader` with 782 batches!

<br>

## **Train**

I created optimizer and scheduler use by PyTorch in training. I used most common parameters used by transformers models.

I looped through the number of defined epochs and call the **train** and **validation** functions.

I'm trying to output similar info after each epoch as Keras: *train_loss:  - val_loss:  - train_acc: - valid_acc*.

After training, plot train and validation loss and accuracy curves to check how the training went.

**Note:** *The training plots might look a little weird: The validation accuracy starts higher than training accuracy and the validation loss starts lower than the training loss. Normally this will be the opposite. I assume the data split just happen to be easier for the validation part or too hard for training part or both. Since this tutorial is about using GPT2 for classification I will not worry about the results of the model too much.*


```python
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
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
    100%|████████████████████████████████|4/4 [15:11<00:00, 227.96s/it]
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:42<00:00, 4.82it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [02:07<00:00, 6.13it/s]
    
      train_loss: 0.54128 - val_loss: 0.38758 - train_acc: 0.75288 - valid_acc: 0.81904
    
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:36<00:00, 5.00it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [01:41<00:00, 7.68it/s]
    
      train_loss: 0.36716 - val_loss: 0.37620 - train_acc: 0.83288 - valid_acc: 0.82912
    
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:36<00:00, 5.00it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [01:24<00:00, 9.24it/s]
    
      train_loss: 0.31409 - val_loss: 0.39384 - train_acc: 0.86304 - valid_acc: 0.83044
    
    
    Training on batches...
    100%|████████████████████████████████|782/782 [02:36<00:00, 4.99it/s]
    
    Validation on batches...
    100%|████████████████████████████████|782/782 [01:09<00:00, 11.29it/s]
    
      train_loss: 0.27358 - val_loss: 0.39798 - train_acc: 0.88432 - valid_acc: 0.83292


![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA1kAAAGNCAYAAADjMbsjAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeXhU1f3H8fe5M5N9IwkhJKyCKC6IEQQVBTFY9+1nC2oRd3EDKiAgChSFsLlUBW2RFsW9VEvr0lag1baorVJFat1FBISQhIQl68w9vz8mDEQIiybcZPJ5PQ+Pztx7J9/JYZlPzrnfY6y1FhEREREREWkQjtcFiIiIiIiIRBOFLBERERERkQakkCUiIiIiItKAFLJEREREREQakEKWiIiIiIhIA1LIEhERERERaUAKWSIiIiIiIg3I73UBjWHDhg1elxCRmZlJUVGR12VII9IYRz+NcfTTGEc/jXH00xhHv6Y4xjk5OXt9vlFD1tq1a5k7dy4VFRXk5uYyYsQI4uPj65wzZcoUiouLiYuLA2Do0KH06NGDwsJCRo4cSbt27QCIjY3l3nvvbcxyRUREREREfrBGDVnz589n8ODB5OXl8dRTT7FkyRKGDBmyx3nDhw/n6KOP3uP59PR0Zs+e3ZglioiIiIiINKhGuyertLSUwsJC8vLyABg4cCDvvPNOY305ERERERGRJqHRZrJKSkrIyMiIPM7MzKS4uHiv5y5YsABjDN27d+eKK66ILCksLS1l3LhxOI7D2WefzWmnnbbX65cuXcrSpUsBmDFjBpmZmQ38br4/v9/fpOqRhqcxjn4a4+inMY5+GuPo19LH2FpLSUkJwWDQ61IaTWFhIdbaQ/51/X4/6enpGGMO/JrGKuZAvwG33XYbGRkZBINBFi5cyKJFi7jhhhto1aoVjz76KCkpKWzevJl77rmHNm3acMQRR+zxGvn5+eTn50ceN6Ub4priDXrSsDTG0U9jHP00xtFPYxz9WvoYV1RUEAgE8Pujsq8dEA47XoTImpoa1q1bt0dvCai/8UWjLRfMyMioM3NVVFRUZ2Zr9/Mg/E0788wz+eSTTwAIBAKkpKQA0Lp1a3r16sWnn37aWOWKiIiIiDRbrutGdcDykt/vx3Xdg7qm0UJWWloaWVlZrFy5EoDly5dz4okn1jknFApRVlYWefzWW2/RsWNHAMrKygiFQgBs376dVatWRY6JiIiIiMguB7OUTQ7ewX5/GzXuXnfddcydO5eFCxeSk5PDiBEjKCkpoaCggNmzZ1NTU0NBQQHBYBBrLbm5uVx77bUAfPzxx7zwwgs4joPrugwYMIAePXo0ZrkiIiIiIiI/mLFe3D3WyLQZsRxKGuPopzGOfhrj6Kcxjn4tfYzLy8tJSEjwuoxG5dU9WVD/9/eQ35MlIiIiIiItz3333fe9rhs6dCgbN278Xtf26dOHb7755ntd2xh0d5yIiIiISBRxn5uP/earBn9d074zzpDr93ve/fffz+jRo/d4PhgM7rM5x6JFi35QfU2JQpaIiIiIiDSIyZMnAzBo0CCSkpLw+XwcddRRvPfee/Ts2ZPBgwdz9913U1lZSSgUYuLEiZx++ulAeDZq8eLFtG/fnj59+nDppZfyt7/9jS1btjBlyhTOOeecA6rhX//6F5MnT6ampobs7GzmzJlDdnY2//rXv5g0aRKhUIhQKMScOXPo2bMn48eP591338VxHI455hgefPDBH/x9UMhqRHbrFmq2l0JSmteliIiIiEgLcSCzTY3l5z//OY8//jivv/46AJdeeiklJSW8/PLLGGPYtm0bixcvJhAIsH79ei6++GLeeeedvXbv8/v9vPLKK7z77ruMGjXqgEJWVVUVN910EwsWLKBnz5489thj3H333cyfP5958+Zx77330qtXL4LBIFVVVfz3v/9l3bp1LF++HIDS0tIG+T7onqxG5C56lC0ThmNXv+d1KSIiIiIinrjooosiIWrHjh3ceuutDBw4kKuuuorCwkI2b9681+vOP/98AI4//njWrl17QF/r888/JyMjg549ewJw2WWXsWLFCgD69u3LlClTeOyxx/jqq69ITEykQ4cOrF+/nokTJ/Lqq68SGxv7Q98uoJDVqJzLbsDXNhf34Xtw//aa1+WIiIiIiBxyu3flmzlzJkceeSTLli3j9ddfJzExkaqqqr1etzPw+Hy+yP65+/PdGbHdHw8fPpwHH3yQmJgYrr/+el566SVSU1P505/+xIABA3jzzTc599xzD/hr7YtCViMy6Zm0mjYPjs7DPv0o7m9/jT3I3aJFRERERJqTpKQktm3bttdj27ZtIycnB2MML7/8coMtz9upS5cuFBcXs2rVKgCee+45Tj75ZAC+/PJL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)

![](data:image/png;base64,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13U1Z84cXXzxxerdu/dx97MvR76fmrHEvuwXjuOoU6dOWrt2baXv16lTR8XFxSoqKpIkFRYWqri4mKNYEehkM5Z+2I+joqLUvXt39uMIkZ2drT179uj222/XyJEjJZVfjOjgwYMV20TKazElqwrEx8erQYMG2rx5syRp1apVSk1N9TgVgqmwsFBHjx6VJJWVlSkrK0tNmjTxOBVO18KFCxUXF3fSU0/YlyPfT82YfTmyHT58WAUFBRW3s7Ky1KhRo0rbGGPUrl07rVmzRpL07rvvql27dlUZE2fhVGZcVlam7777ruL2hg0b2I8jRPfu3fX4448rEAgoEAhIkgKBQKVTByPltZgPIw6CBQsWaMuWLcrPz1f9+vWVnJys66+/XkuWLNE999wjqfzwZyAQUGFhoRo2bKj09PTjLjeK8PVTM963b58eeughWWvluq6aN2+uoUOHVpzzj/D32WefaeLEiWrUqJEcp/zfn7p06aIWLVqwL/vEqcyYfTmy7du3T7Nnz1ZJSYmstUpKStLvfvc7ua6rqVOnVqy1y83N1Zw5cypdwj0c/yUcxzuVGRcWFmry5MkqLS2t2ObWW2/Vueee63V8nKZbbrlFS5Ys0RdffBFxr8WULAAAAAAIIk4XBAAAAIAgomQBAAAAQBBRsgAAAAAgiChZAAAAABBElCwAAAAACCJKFgAgYhljlJycXPHfwIEDg/4z1qxZo86dOwf9eQEA/hXldQAAAM7Gli1bvI4AAEAlHMkCAPhOZmamevbsqe7du6tly5YaPHiwiouLJUl79uxRjx491KZNG11xxRVav359xeNefvllXX755Wrbtq3at2+voqIiSVJhYaEGDRqkSy+9VB07dlRubq4k6bXXXlObNm2UnJysNm3aaNeuXVX/ywIAwg4lCwAQ0f79dMEJEyZUfP/999/X008/rR07dujIkSNasGCBJGnUqFHq3r27Pv74Y82fP1/9+/dXUVGRduzYobvuuktvvfWWtm7dqrfffls1a9aUJG3dulX33Xeftm3bplatWmnhwoWSpEmTJmn58uXasmWLsrKydP7551f9HwAAIOxwuiAAIKKd7HTBrl27KikpSZI0ePBgPfPMM0pPT9eaNWuUmZkpSbryyiuVkJCg7OxsrV27Vr/+9a8rHlOvXr2K50pOTlbz5s0lSampqdqwYYMkqXPnzhoyZIj69OmjXr16qUmTJqH6NQEAEYQjWQCAasUY86O3TyQmJqbi6xo1aqi0tFSSNGfOHE2fPl1HjhxR586dtXbt2uCGBQBEJEoWAMCXVq9erX/961+y1uqFF15Qly5dJJUffXr66aclSZs2bVJeXp5atGihbt266fXXX9fevXslSQUFBXJd90d/xueff67LL79c48aN0/XXX6+PPvootL8UACAicLogACCiJScnV3x9/vnna/ny5ZKkq6++WkOGDNHu3buVmpqqYcOGSSo/+pSWlqannnpK0dHRWrx4sWJiYtSyZUs9/PDD6tmzpySpdu3aWr169Y/+7HHjxmnnzp2KiopS48aNNWPGjBD9lgCASGKstdbrEAAABFNmZmaltVcAAFQlThcEAAAAgCDiSBYAAAAABBFHsgAAAAAgiChZAAAAABBElCwAAAAACCJKFgAAAAAEESULAAAAAIKIkgUAAAAAQfT/ALjNz+I7kJUDAAAAAElFTkSuQmCC)


## **Evaluate**

When dealing with classification is useful to look at precision recall and F1 score.

A good gauge to have when evaluating a model is the confusion matrix.


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
                      magnify=0.1,
                      );
```

    Training on batches...
    100%|████████████████████████████████|782/782 [01:09<00:00, 11.24it/s]
    
                  precision    recall  f1-score   support

             neg       0.84      0.83      0.83     12500
             pos       0.83      0.84      0.83     12500
    
        accuracy                           0.83     25000
       macro avg       0.83      0.83      0.83     25000
    weighted avg       0.83      0.83      0.83     25000

![](data:image/png;base64,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)

<br>

## **Final Note**

If you made it this far **Congrats!** 🎊 and **Thank you!** 🙏 for your interest in my tutorial!

I've been using this code for a while now and I feel it got to a point where is nicely documented and easy to follow.

Of course is easy for me to follow because I built it. That is why any feedback is welcome and it helps me improve my future tutorials!

If you see something wrong please let me know by opening an issue on my [ml_things GitHub repository](https://github.com/gmihaila/ml_things/issues)!

A lot of tutorials out there are mostly a one-time thing and are not being maintained. I plan on keeping my tutorials up to date as much as I can.

## **Contact** 🎣

🦊 GitHub: [gmihaila](https://github.com/gmihaila)

🌐 Website: [gmihaila.github.io](https://gmihaila.github.io/)

👔 LinkedIn: [mihailageorge](https://www.linkedin.com/in/mihailageorge/)

📓 Medium: [@gmihaila](https://gmihaila.medium.com)

📬 Email: [georgemihaila@my.unt.edu.com](mailto:georgemihaila@my.unt.edu.com?subject=GitHub%20Website)

<br>