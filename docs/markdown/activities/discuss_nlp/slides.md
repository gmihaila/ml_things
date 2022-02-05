---
title: Discuss NLP & 🤗 Library
description: Discussion around my research in NLP and around the HuggingFace library.
theme: default
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.jpg')
size: 5:4
footer: '2021 George Mihaila'
---

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# <!-- fit --> Discuss NLP & 🤗 Library

<br/><br/>

**George Mihaila**

*PhD Candidate Computer Science*
*University of North Texas*

<br/>

---
# Disclaimer

- This is not a thorough tutorial on NLP.
- This is not a thorough tutorial on the Hugging Face Library.
- This presentation is to encourage curiosity and further exploration of NLP and the Hugging Face Library.

---
# Agenda

* About me 💬
* Natural Language Processing 📖
* Chatbots with Attitude 🗣️
* Topic Shift Detection 🕵️‍
* Identify Innovation 💡
* 🤔 🤗 ⚙️
* My Tutorials 📚
* Conclusions 🤔
* Questions 🖐
* Contact 🎣

<br>


---
# About me 💬

* PhD candidate in computer science at University of North Texas (UNT).
* Research area in Natural Language Processing (NLP) with focus on dialogue generation with persona.
* Four years of combined experience in research and industry on various Artificial Intelligence (AI) and Machine Learning (ML) projects. Check my resume [here](https://gmihaila.github.io/resume/resume/).
* **Interest areas:** `Neural Networks`, `Deep Learning`, `Natural Language Processing`, `Reinforcement Learning`, `Computer Vision`,
`Scaling Machine Learning`.


---
# About me 💬

## How I got here?

* Neural networks are the reason I started my PhD in computer science. 
* The professor I talked with asked me if I wanted to work on natural language processing and neural networks. 
* The notion of neural networks sounded very exciting to me. 
* That’s when I knew **that** is what I want to do for my career.


---
# About me 💬

## In my free time

* I like to share my knowledge on NLP: I wrote tutorials from scratch on state-of-the-art language models like Bert and GPT2 with over 10k views. Check them out [here](https://gmihaila.medium.com).
* Contribute to open-source – Hugging Face `Transformers` and `Datasets`.
* Technical reviewer for one of the first books published on transformers models for NLP. The book is called **Transformers for NLP in Python**, by *Denis Rothman*.
* Technical reviewer for the next edition of the **Transformers for NLP in Python** book.
* I delivered a webinar on my tutorial **GPT2 for Text Classification** with 300+ participants.
* My personal project [ML Things](https://github.com/gmihaila/ml_things) for things I find useful and speed up my work with ML. 


---
# Natural Language Processing 📖

## Wikipedia

**Natural language processing (NLP)** is a subfield of linguistics, computer science, and artificial intelligence **concerned with the interactions between computers and human language** ...

... The result is a computer capable of **"understanding"** the contents of documents, including the contextual nuances of the language within them.


---
# Chatbots with Attitude 🗣️

## Research

* My main research area is in NLP with focus on dialogue generation with persona. 
* I use the Friends TV corpus to train language models that can capture each of the main six characters personas. 
* Imagine a chatbot that sounds like Joey or Chandler. 
* This will make chatbots systems more engaging and allow shifting personas depending on a customer's mood.
* It can significantly increase customer experience.


---
# Chatbots with Attitude 🗣️

## Data

* Create dataset around each of the six main character in the form of `context - response` pairs from all ten seasons.
* **Context:** dialogue history of target character or other characters.
* **Response:** sentence that target character responds.
* Train data: `Season 1 - Season 8`
* Validation data: `Season 9`
* Test data: `Season: 10`


---
# Chatbots with Attitude 🗣️

## Model

* Use GPT-2 to generate responses as baseline model.
* Fine-tune GPT-2 on each of the six main characters dataset (six different GPT-2 models)
* Use special token separator between context and response `<SPEAKER>`.


---
# Chatbots with Attitude 🗣️

## Evaluation

* Use Bilingual Evaluation Understudy (BLEU) score to evaluate model performance (higher is better).
* Evaluate each of the six models on each of the six characters validation data.
* Each of the six models should have the highest BLEU score on their target character's validation data and lower on other character's validation data.

---
# Topic Shift Detection 🕵️‍

* Detect sudden topic shift in dialogue conversation.
* Help determine when a conversation changes topic and find out which topic is a speaker more interested in.
* Use similar `context-response` format data to classify if the response is in a different topic or not.

---
# Identify Innovation 💡

* Use patent data to automatically detect fintech innovations.
* *Identifying FinTech Innovations Using BERT* full paper published in **IEEE Big Data 2020**.
* Classify a patent abstract into six types of fintech categories. Check more [here](https://github.com/gmihaila/fintech_patents).

---
# 🤔 🤗 ⚙️

## About

* Hugging Face is a company that offer state-of-the art models and solutions in NLP.
* It's based on open-source.
* Was founded in 2016 by Clement Delangue, Julien Chaumond, Thomas Wolf.
* Contains thousands of language models on 22 tasks.
* Most model implementations come in TensorFlow 2 and PyTorch.
* Contains 449+ datasets on various NLP tasks.
* Let's check it out [huggingface.co](https://huggingface.co).


---
# 🤔 🤗 ⚙️

## Transformers

* Library dedicated to transformers models architectures.
* Contains models implementation, tokenizers and pre-trained weights from published papers.
* Let's check it out [huggingface.co/transformers](https://huggingface.co/transformers/quicktour.html)
* Let's check their GitHub out [github.com/huggingface/transformers](https://github.com/huggingface/transformers)

---
# 🤔 🤗 ⚙️

## Datasets

* Library dedicated to popular NLP datasets for various tasks: summarization, sentiment, dialogue, etc.
* Contains efficient loading and formatting of large datasets.
* Let's check their documentation out [huggingface.co/docs/datasets/](https://huggingface.co/docs/datasets/)

---
# My Tutorials 📚

| Name 	| Description 	| Links 	|
|:- |:- |:- |
| **:grapes: Better Batches with PyTorchText BucketIterator** | *How to use PyTorchText BucketIterator to sort text data for better batching.* |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/pytorchtext_bucketiterator.ipynb) [![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/pytorchtext_bucketiterator.ipynb) [![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/7gyq6qup6y43z9b/pytorchtext_bucketiterator.ipynb?dl=1) [![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://gmihaila.medium.com/better-batches-with-pytorchtext-bucketiterator-12804a545e2a) [![Generic badge](https://img.shields.io/badge/Blog-Post-blue.svg)](https://gmihaila.github.io/tutorial_notebooks/pytorchtext_bucketiterator/) |
| **:dog: Pretrain Transformers Models in PyTorch using Hugging Face Transformers** | *Pretrain 67 transformers models on your custom dataset.* |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/pretrain_transformers_pytorch.ipynb) [![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/pretrain_transformers_pytorch.ipynb) [![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/rkq79hwzhqa6x8k/pretrain_transformers_pytorch.ipynb?dl=1) [![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://gmihaila.medium.com/pretrain-transformers-models-in-pytorch-using-transformers-ecaaec00fbaa) [![Generic badge](https://img.shields.io/badge/Blog-Post-blue.svg)](https://gmihaila.github.io/tutorial_notebooks/pretrain_transformers_pytorch/) |
| **:violin: Fine-tune Transformers in PyTorch using Hugging Face Transformers** | *Complete tutorial on how to fine-tune 73 transformer models for text classification — no code changes necessary!* |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/finetune_transformers_pytorch.ipynb) [![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/finetune_transformers_pytorch.ipynb) [![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/tsqicfqgt8v87ae/finetune_transformers_pytorch.ipynb?dl=1) [![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://medium.com/@gmihaila/fine-tune-transformers-in-pytorch-using-transformers-57b40450635) [![Generic badge](https://img.shields.io/badge/Blog-Post-blue.svg)](https://gmihaila.github.io/tutorial_notebooks/finetune_transformers_pytorch/)|
| **⚙️ Bert Inner Workings in PyTorch using Hugging Face Transformers** | *Complete tutorial on how an input flows through Bert.* |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/bert_inner_workings.ipynb) [![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/bert_inner_workings.ipynb) [![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/jeftyo6cebfkma2/bert_inner_workings.ipynb?dl=1) [![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://gmihaila.medium.com/%EF%B8%8F-bert-inner-workings-1c3054cd1591) [![Generic badge](https://img.shields.io/badge/Blog-Post-blue.svg)](https://gmihaila.github.io/tutorial_notebooks/bert_inner_workings/)|
| **🎱 GPT2 For Text Classification using Hugging Face 🤗 Transformers** | *Complete tutorial on how to use GPT2 for text classification.* |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb) [![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/gpt2_finetune_classification.ipynb) [![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/6t6kvlewoabwxqw/gpt2_finetune_classification.ipynb?dl=1) [![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://gmihaila.medium.com/gpt2-for-text-classification-using-hugging-face-transformers-574555451832) [![Generic badge](https://img.shields.io/badge/Blog-Post-blue.svg)](https://gmihaila.github.io/tutorial_notebooks/gpt2_finetune_classification/)|


---
# Conclusions 🤔

* You learned a little bit about myself.
* Learned more about **Hugging Face**.
* Learned how to navigate the **Transformers** library.
* Know about the **Datasets** library.

---
# Questions 🖐

- What did you learn today?
- What motivated you in this presentation?
- Do you have any questions?

---
# Contact 🎣

Let's stay in touch!


🦊 GitHub: [gmihaila](https://github.com/gmihaila)

🌐 Website: [gmihaila.github.io](https://gmihaila.github.io/)

👔 LinkedIn: [mihailageorge](https://www.linkedin.com/in/mihailageorge/)

📓 Medium: [@gmihaila](https://gmihaila.medium.com)

📬 Email: [georgemihaila@my.unt.edu.com](mailto:georgemihaila@my.unt.edu.com?subject=GitHub%20Website)

👤 Schedule meeting: [calendly.com/georgemihaila](https://calendly.com/georgemihaila)


