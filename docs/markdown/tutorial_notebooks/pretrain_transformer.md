# :dog: Pretrain Transformers


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14KCDms4YLrE7Ekxl9VtrdT229UTDyim3#offline=true&sandboxMode=true)
[![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/machine_learning_things/blob/master/tutorial_notebooks/pretrain_transformer.ipynb)
[![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://dl.dropbox.com/s/ihtxu3k70mj37pj/pretrain_transformer.ipynb?dl=1)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Info

This notebook is used to pretrain transformers models using [Huggingface](https://huggingface.co/transformers/). This notebooks is part of my trusty notebooks for Machine Learning. Check out more similar content on my website [gmihaila.github.io/useful/useful/](https://gmihaila.github.io/useful/useful/) where I post useful notebooks like this one.

This notebook is **heavily inspired** from the Huggingface script used for training language models: [transformers/tree/master/examples/language-modeling](https://github.com/huggingface/transformers/tree/master/examples/language-modeling).

'Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.'

<br>

## How to use this notebook? 

This notebooks is a code adaptation of the [run_language_modeling.py](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py). 

**Models that are guarantee to work:** [GPT](https://huggingface.co/transformers/model_summary.html#original-gpt), [GPT-2](https://huggingface.co/transformers/model_summary.html#gpt-2), [BERT](https://huggingface.co/transformers/model_summary.html#bert), [DistilBERT](https://huggingface.co/transformers/model_summary.html#distilbert), [RoBERTa](https://huggingface.co/transformers/model_summary.html#roberta) and [XLNet](https://huggingface.co/transformers/model_summary.html#xlnet). 

Parse the arguments needed that are split in TrainingArguments, ModelArguments and DataTrainingArguments. The only variables that need configuration depending on your needs are `model_args`, `data_args` and `training_args` in **Parameters**:

* `model_args` of type **ModelArguments**: These are the arguments for the model that you want to use such as the model_name_or_path, tokenizer_name etc. You'll need these to load the model and tokenizer.

  Minimum setup:

  ```python
  model_args = ModelArguments(model_name_or_path, 
                            model_type,
                            tokenizer_name,
                            )
  ```

  * `model_name_or_path` path to existing transformers model or name of transformer model to be used: *bert-base-cased*, *roberta-base*, *gpt2* etc. More details [here](https://huggingface.co/transformers/pretrained_models.html).

  * `model_type` type of model used: *bert*, *roberta*, *gpt2*. More details [here](https://huggingface.co/transformers/pretrained_models.html).

  * `tokenizer_name` [tokenizer](https://huggingface.co/transformers/main_classes/tokenizer.html#tokenizer) used to process data for training the model. It usually has same name as `model_name_or_path`: *bert-base-cased*, *roberta-base*, *gpt2* etc.


* `data_args` of type **DataTrainingArguments**: These are as the name suggests arguments needed for the dataset. Such as the directory name where your files are stored etc. You'll need these to load/process the dataset.

  Minimum setup:

  ```python
  data_args = DataArgs(train_data_file,
                     eval_data_file,
                     mlm,
                     )
  ```
  
  * `train_data_file` path to your dataset. This is a plain file that contains all your text data to train a model. Use each line to separate examples: i.e. if you have a dataset composed of multiple  text documents, create a single file with each line in the file associated to a text document.

  * `eval_data_file` same story as `train_data_file`. This file is used to evaluate the model performance

  * `mlm` is a flag that changes loss function depending on model architecture. This variable needs to be set to **True** when working with masked language models like *bert* or *roberta*.



* `training_args` of type **TrainingArguments**: These are the training hyper-parameters such as learning rate, batch size, weight decay, gradient accumulation steps etc. See all possible arguments [here](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py). These are used by the Trainer.

  Minimum setup:

* `model_args`
  ```python
  training_args = TrainingArguments(output_dir, 
                                  do_train, 
                                  do_eval,
                                  )
  ```

  * `output_dir` path where to save the pre-trained model.
  * `do_train` variable to signal if you're using train data or not. Set it to **True** if you mentioned `train_data_file`.
  * `do_eval` variable to signal if you're using evaluate data or not. Set it to **True** if you mentioned `eval_data_file`.

<br>

## Example:

### Pre-train Bert

In the **Parameters** section use arguments:

```python
# process model arguments. Check Info - Notes for more details
model_args = ModelArguments(model_name_or_path='bert-base-cased', 
                            model_type='bert',
                            tokenizer_name='bert-base-cased',
                            )

# process data arguments. Check Info - Notes for more details
data_args = DataArgs(train_data_file='/content/your_train_data',
                     eval_data_file='/content/your_test_data,
                     mlm=True,
                     )

# process training arguments. Check Info - Notes for more details
training_args = TrainingArguments(output_dir='/content/pretrained_bert', 
                                  do_train=True, 
                                  do_eval=False)
```


<br>

## Notes:
* Parameters details got from [here](https://github.com/patil-suraj/exploring-T5/blob/master/T5_on_TPU.ipynb).

* **Models that are guarantee to work:** [GPT](https://huggingface.co/transformers/model_summary.html#original-gpt), [GPT-2](https://huggingface.co/transformers/model_summary.html#gpt-2), [BERT](https://huggingface.co/transformers/model_summary.html#bert), [DistilBERT](https://huggingface.co/transformers/model_summary.html#distilbert), [RoBERTa](https://huggingface.co/transformers/model_summary.html#roberta) and [XLNet](https://huggingface.co/transformers/model_summary.html#xlnet). I plan on testing more models in the future.
* I used the [The WikiText Long Term Dependency Language Modeling Dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) as an example. **To reduce training time I used the evaluate split as training and test split as evaluation!**.
