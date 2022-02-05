# :grapes: **Better Batches with PyTorchText BucketIterator**

## **How to use PyTorchText BucketIterator to sort text data for better batching.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gmihaila/ml_things/blob/master/notebooks/pytorch/pytorchtext_bucketiterator.ipynb) &nbsp;
[![Generic badge](https://img.shields.io/badge/GitHub-Source-greensvg)](https://github.com/gmihaila/ml_things/blob/master/notebooks/pytorch/pytorchtext_bucketiterator.ipynb)
[![Generic badge](https://img.shields.io/badge/Download-Notebook-red.svg)](https://www.dropbox.com/s/7gyq6qup6y43z9b/pytorchtext_bucketiterator.ipynb?dl=1)
[![Generic badge](https://img.shields.io/badge/Article-Medium-black.svg)](https://gmihaila.medium.com/better-batches-with-pytorchtext-bucketiterator-12804a545e2a)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<br>

**Disclaimer:** *The format of this tutorial notebook is very similar with my other tutorial notebooks. This is done intentionally in order to keep readers familiar with my format.*

<br>

This notebook is a simple tutorial on how to use the powerful **PytorchText**  **BucketIterator** functionality to group examples (**I use examples and sequences interchangeably**) of similar lengths into batches. This allows us to provide the most optimal batches when training models with text data.

Having batches with similar length examples provides a lot of gain for recurrent models (RNN, GRU, LSTM) and transformers models (bert, roBerta, gpt2, xlnet, etc.) where padding will be minimal.

Basically any model that takes as input variable text data sequences will benefit from this tutorial.

**I will not train any models in this notebook!** I will release a tutorial where I use this implementation to train a transformer model.

The purpose is to use an example text datasets and batch it using **PyTorchText** with **BucketIterator** and show how it groups text sequences of similar length in batches.

This tutorial has two main parts:

* **Using PyTorch Dataset with PyTorchText Bucket Iterator**: Here I implemented a standard PyTorch Dataset class that reads in the example text datasets and use PyTorch Bucket Iterator to group similar length examples in same batches. I want to show how easy it is to use this powerful functionality form PyTorchText on a regular PyTorch Dataset workflow which you already have setup.

* **Using PyTorch Text TabularDataset with PyTorchText Bucket Iterator**: Here I use the built-in PyTorchText TabularDataset that reads data straight from local files without the need to create a PyTorch Dataset class. Then I follow same steps as in the previous part to show how nicely text examples are grouped together.

*This notebooks is a code adaptation and implementation inspired from a few sources:* [torchtext_translation_tutorial](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html), [pytorch/text - GitHub](https://github.com/pytorch/text), [torchtext documentation](https://torchtext.readthedocs.io/en/latest/index.html#) and [A Comprehensive Introduction to Torchtext](https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/).

<br>

## **What should I know for this notebook?**
Some basic PyTorch regarding Dataset class and using DataLoaders. Some knowledge of PyTorchText is helpful but not critical in understanding this tutorial. The BucketIterator is similar in applying Dataloader to a PyTorch Dataset.

<br>

## **How to use this notebook?**
The code is made with reusability in mind. It can be easily adapted for other text datasets and other NLP tasks in order to achieve optimal batching.

Comments should provide enough guidance to easily adapt this notebook to your needs.

This code is designed mostly for **classification tasks** in mind, but it can be adapted for any other Natural Language Processing tasks where batching text data is needed.


<br>

## **Dataset**

I will use the well known movies reviews positive - negative labeled [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

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

Download the IMDB Movie Reviews sentiment dataset and unzip it locally.


```
# download the dataset
!wget -q -nc http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# unzip it
!tar -zxf /content/aclImdb_v1.tar.gz
```

## **Installs**

* **[ml_things](https://github.com/gmihaila/ml_things)** library used for various machine learning related tasks. I created this library to reduce the amount of code I need to write for each machine learning project.



```
# Install helper functions.
!pip install -q git+https://github.com/gmihaila/ml_things.git
```

    |████████████████████████████████| 71kB 5.2MB/s
    Building wheel for ml-things (setup.py) ... done
    Building wheel for ftfy (setup.py) ... done


## **Imports**

Import all needed libraries for this notebook.

Declare basic parameters used for this notebook:

* `device` - Device to use by torch: GPU/CPU. I use CPU as default since I will not perform any costly operations.

* `train_batch_size` - Batch size used on train data.

* `valid_batch_size` - Batch size used for validation data. It usually is greater than `train_batch_size` since the model would only need to make prediction and no gradient calculations is needed.


```python
import io
import os
import torchtext
from tqdm.notebook import tqdm
from ml_things import fix_text
from torch.utils.data import Dataset, DataLoader

# Will use `cpu` for simplicity.
device = 'cpu'

# Number of batches for training
train_batch_size = 10

# Number of batches for validation. Use a larger value than training.
# It helps speed up the validation process.
valid_batch_size = 20
```

## **Using PyTorch Dataset**

This is where I create the PyTorch Dataset objects for training and validation that **can** be used to feed data into a model. This is standard procedure when using PyTorch.



### **Dataset Class**

Implementation of the PyTorch Dataset class.

Most important components in a PyTorch Dataset class are:

* `__len__(self, )` where it returns the number of examples in our dataset that we read in `__init__(self, )`. This will ensure that `len()` will return the number of examples.

* `__getitem__(self, item)` where given an index `item` will return the example corresponding to the `item` position.


```python
class MovieReviewsTextDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self, path):

    # Check if path exists.
    if not os.path.isdir(path):
      # Raise error if path is invalid.
      raise ValueError('Invalid `path` variable! Needs to be a directory')

    self.texts = []
    self.labels = []
    # Since the labels are defined by folders with data we loop
    # through each label.
    for label  in ['pos', 'neg']:
      sentiment_path = os.path.join(path, label)

      # Get all files from path.
      files_names = os.listdir(sentiment_path)#[:10] # Sample for debugging.
      # Go through each file and read its content.
      for file_name in tqdm(files_names, desc=f'{label} Files'):
        file_path = os.path.join(sentiment_path, file_name)

        # Read content.
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        # Fix any unicode issues.
        content = fix_text(content)
        # Save content.
        self.texts.append(content)
        # Save labels.
        self.labels.append(label)

    # Number of examples.
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
      :obj:`Dict[str, str]`: Dictionary of inputs that are used to feed
      to a model.

    """

    return {'text':self.texts[item], 'label':self.labels[item]}
```

### **Train - Validation Datasets**

Create PyTorch Dataset for train and validation partitions.


```python
print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = MovieReviewsTextDataset(path='/content/aclImdb/train')

print(f'Created `train_dataset` with {len(train_dataset)} examples!')

print()

print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset =  MovieReviewsTextDataset(path='/content/aclImdb/test')

print(f'Created `valid_dataset` with {len(valid_dataset)} examples!')
```

    Dealing with Train...
    pos Files: 100% |████████████████████████████████| 12500/12500 [01:22<00:00, 151.34it/s]
    neg Files: 100% |████████████████████████████████| 12500/12500 [01:10<00:00, 178.52it/s]
    Created `train_dataset` with 25000 examples!

    Dealing with Validation...
    pos Files: 100% |████████████████████████████████| 12500/12500 [01:22<00:00, 151.34it/s]
    neg Files: 100% |████████████████████████████████| 12500/12500 [01:10<00:00, 178.52it/s]
    Created `valid_dataset` with 25000 examples!


### **PyTorch DataLoader**

In order to group examples from the PyTorch Dataset into batches we use PyTorch DataLoader. This is standard when using PyTorch.


```python
# Move pytorch dataset into dataloader.
torch_train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
print(f'Created `torch_train_dataloader` with {len(torch_train_dataloader)} batches!')

# Move pytorch dataset into dataloader.
torch_valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)
print(f'Created `torch_valid_dataloader` with {len(torch_valid_dataloader)} batches!')
```

    Created `torch_train_dataloader` with 2500 batches!
    Created `torch_valid_dataloader` with 1250 batches!


### **PyTorchText Bucket Iterator Dataloader**

Here is where the magic happens! We pass in the **train_dataset** and **valid_dataset** PyTorch Dataset splits into **BucketIterator** to create the actual batches.

It's very nice that PyTorchText can handle splits! No need to write same line of code again for train and validation split.

**The `sort_key` parameter is very important!** It is used to order text sequences in batches. Since we want to batch sequences of text with similar length, we will use a simple function that returns the length of an data example (`len(x['text')`). This function needs to follow the format of the PyTorch Dataset we created in order to return the length of an example, in my case I return a dictionary with `text` key for an example.

**It is important to keep `sort=False` and `sort_with_batch=True` to only sort the examples in each batch and not the examples in the whole dataset!**

Find more details in the PyTorchText **BucketIterator** documentation [here](https://torchtext.readthedocs.io/en/latest/data.html#bucketiterator) - look at the **BPTTIterator** because it has same parameters except the **bptt_len** argument.

**Note:**
*If you want just a single DataLoader use `torchtext.data.BucketIterator` instead of `torchtext.data.BucketIterator.splits` and make sure to provide just one PyTorch Dataset instead of tuple of PyTorch Datasets and change the parameter `batch_sizes` and its tuple values to `batch_size` with single value: `dataloader = torchtext.data.BucketIterator(dataset, batch_size=batch_size, )`*


```python
# Group similar length text sequences together in batches.
torchtext_train_dataloader, torchtext_valid_dataloader = torchtext.data.BucketIterator.splits(

                              # Datasets for iterator to draw data from
                              (train_dataset, valid_dataset),

                              # Tuple of train and validation batch sizes.
                              batch_sizes=(train_batch_size, valid_batch_size),

                              # Device to load batches on.
                              device=device,

                              # Function to use for sorting examples.
                              sort_key=lambda x: len(x['text']),


                              # Repeat the iterator for multiple epochs.
                              repeat=True,

                              # Sort all examples in data using `sort_key`.
                              sort=False,

                              # Shuffle data on each epoch run.
                              shuffle=True,

                              # Use `sort_key` to sort examples in each batch.
                              sort_within_batch=True,
                              )

# Print number of batches in each split.
print('Created `torchtext_train_dataloader` with %d batches!'%len(torchtext_train_dataloader))
print('Created `torchtext_valid_dataloader` with %d batches!'%len(torchtext_valid_dataloader))
```

    Created `torchtext_train_dataloader` with 2500 batches!
    Created `torchtext_valid_dataloader` with 1250 batches!


### **Compare DataLoaders**

Let's compare the PyTorch DataLoader batches with the PyTorchText BucketIterator batches. We can see how nicely examples of similar length are grouped in same batch with PyTorchText.

**Note:** *When using the PyTorchText BucketIterator, make sure to call `create_batches()` before looping through each batch! Else you won't get any output form the iterator.*


```python
# Loop through regular dataloader.
print('PyTorch DataLoader\n')
for batch in torch_train_dataloader:

  # Let's check batch size.
  print('Batch size: %d\n'% len(batch['text']))
  print('LABEL\tLENGTH\tTEXT'.ljust(10))

  # Print each example.
  for text, label in zip(batch['text'], batch['label']):
    print('%s\t%d\t%s'.ljust(10) % (label, len(text), text))
  print('\n')

  # Only look at first batch. Reuse this code in training models.
  break


# Create batches - needs to be called before each loop.
torchtext_train_dataloader.create_batches()

# Loop through BucketIterator.
print('PyTorchText BuketIterator\n')
for batch in torchtext_train_dataloader.batches:

  # Let's check batch size.
  print('Batch size: %d\n'% len(batch))
  print('LABEL\tLENGTH\tTEXT'.ljust(10))

  # Print each example.
  for example in batch:
    print('%s\t%d\t%s'.ljust(10) % (example['label'], len(example['text']), example['text']))
  print('\n')

  # Only look at first batch. Reuse this code in training models.
  break
```

    PyTorch DataLoader

    Batch size: 10

    LABEL LENGTH TEXT
    neg	  811	 Much as we all love Al Pacino, it was painful to see him in this movie. A publicity hack at the grubby ending of what seems to have once been a distinguished and idealistic career Pacino plays his part looking like an unmade bed and assaulting everyone with a totally bogus and inconsistent southern accent.<br /><br />The plot spools out this way and that with so many loose ends and improbabilities that the mind reels (and then retreats).<br /><br />Kim Basinger is there, not doing much. Her scenes with Pacino are flat and unconvincing. Hard to believe they meant a lot to each other. There's no energy there.<br /><br />Tea Leone, on the other hand, lit up the screen. She was electric and her scenes with Pacino were by far the most interesting in the movie, but not enough to save Al from embarrassment.  
    neg	  572	 God, I am so sick of the crap that comes out of America called "Cartoons"!<br /><br />Since anime became popular, the USA animators either produce a cartoon with a 5-year-old-lazy-ass style of drawing (Kappa Mikey) or some cheep Japanese animation rip-off. (Usually messing up classic characters) No wonder anime is beating American cartoons! <br /><br />They are not even trying anymore! <br /><br />Oh, I just heard of this last night; I live in the UK and when I found out this show first came out in 2005,well, I never knew the UK was so up-to-date with current shows.  
    pos	  3122	 First an explanation on what makes a great movie for me. Excitement about not knowing what is coming next will make me enjoy a movie the first time I watch it (case en point: Twister). There are also other things that go into a great first viewing such as good humor (John Candy in Uncle Buck and The Great Outdoors), good plot with good resolution (Madeline and Matilda), imaginative storytelling (all Star Wars episodes-George Lucas is THE MAN), and good music (again all Star Wars episodes, Wizard of Oz, Sound of Music). What makes me watch a movie at least six times in the theatre and buy a DVD or VHS tape? Characters. With that said, I present Cindy Lou Who and The Grinch. Excellent performance Taylor Momsen and Jim Carrey. The rest of the cast was very good, particularly Jeffery Tambor, Bill Irwin, Molly Shannon, Christine Baranski, and Josh Ryan Evans. But, every single scene with Cindy and The Grinch-together is excellent and very funny and/or heartwarming. Cindy Lou is my favorite character in this movie and the most compelling reason why the movie is better than the cartoon. The Grinch has a strong plot, good conflicts, and a very good theme (I can't get started because I don't want to spoil it). Jim Carrey was very funny as The Grinch-particularly when he interacted with Cindy. And the music! Wow! Excellent music by James Horner. I loved his selection of instruments and the compositions. Very good job Jim Carrey-I didn't know you could sing. Taylor Momsen! Whoa! Your voice is reason enough to see the movie at least once. On your solo - Where Are You Christmas - is your voice really as high as it sounds? Sounds like an F#? That is an obscene range for a 7-year old (obscene meant in the best possible way). Great job. This is the best performance by a child I have ever heard in a movie(Taylor beat out the Von Trapp Children-no small feat!). And now to the actors. Jim Carrey was great, funny, and, surprisingly very sensitive (this really showed through in his scenes with Taylor Momsen). Taylor Momsen's unspoken expressions(one of the secrets to a good acting performance) are very strong-she really becomes Cindy Lou Who. And when she does dialogue she is even stronger.<br /><br />******************************danger:spoiler alert********************* ***********************************************************************<br /><br />Examples: expression when she first sees The Grinch. This is a classic quote ("You're the the the" and then filled in with the Grinch line "da da da THE GRINCH-after which she topples into the sorter and then is rescued by The Grinch). The "Thanks for saving me" quote and subsequent response by The Grinch was also very good.<br /><br />My favorite part of the movie is when Cindy invites The Grinch to be Holiday Cheermeister. This scene is two excellent actors at their best interacting and expressing with each other. Little Taylor Momsen completely holds her own with Jim Carrey in this spot. I sincerely hope we see Taylor Momsen in many more films to come. All in all everything was great about this movie (except maybe the feet and noses).  
    pos	  483	 Red Rock West is one of those rare films that keeps you guessing the entire time as to what will happen next. Nicolas Cage is mistaken for a contract killer as he enters a small town trying to find work. Dennis Hopper is the bad guy and no one plays them better. Look for a brief appearance by country singing star Dwight Yoakam. This is a serious drama most of the time but there are some lighter moments. What matters is that you will enjoy this low budget but high quality effort!  
    pos	  759	 This movie is a remake of two movies that were a lot better. The last one, Heaven Can Wait, was great, I suggest you see that one. This one is not so great. The last third of the movie is not so bad and Chris Rock starts to show some of the comic fun that got him to where he is today. However, I don't know what happened to the first two parts of this movie. It plays like some really bad "B" movie where people sound like they are in some bad TV sit-com. The situations are forced and it is like they are just trying to get the story over so they can start the real movie. It all seems real fake and the editing is just bad. I don't know how they could release this movie like that. Anyway, the last part isn't to bad, so wait for the video and see it then.  
    pos	  2471	 VIVAH in my opinion is the best movie of 2006, coming from a director that has proved successful throughout his career. I am not too keen in romantic movies these days, because i see them as "old wine in a new bottle" and so predictable. However, i have watched this movie three times now...and believe me it's an awesome movie.<br /><br />VIVAH goes back to the traditional route, displaying simple characters into a sensible and realistic story of the journey between engagement and marriage. The movie entertains in all manners as it can be reflected to what we do (or would do) when it comes to marriage. In that sense Sooraj R. Barjatya has done his homework well and has depicted a very realistic story into a well-made highly entertaining movie.<br /><br />Several sequences in this movie catch your interest immediately: <br /><br />* When Shahid Kapoor comes to see the bride (Amrita Rao) - the way he tries to look at her without making it too obvious in front of his and her family. The song 'Do Anjaane Ajnabi' goes well with the mood of this scene.<br /><br />* The first conversation between Shahid and Amrita, when he comes to see her - i.e. a shy Shahid not knowing exactly what to talk about but pulling of a decent conversation. Also Amrita's naive nature, limited eye-contact, shy characteristics and answering softly to Shahid's questions.<br /><br />* The emotional breakdown of Amrita and her uncle (Alok Nath) when she feeds him at Shahid's party in the form of another's daughter-in-law rather than her uncle's beloved niece.<br /><br />Clearly the movie belongs to Amrita Rao all the way. The actress portrays the role of Poonam with such conviction that you cannot imagine anybody else replacing her. She looks beautiful throughout the whole movie, and portrays an innocent and shy traditional girl perfectly.<br /><br />Shahid Kapoor performs brilliantly too. He delivers a promising performance and shows that he is no less than Salman Khan when it comes to acting in a Sooraj R. Barjatya film. In fact Shahid and Amrita make a cute on-screen couple, without a shadow of doubt. Other characters - Alok Nath (Excellent), Anupam Kher (Brilliant), Mohan Joshi (Very good).<br /><br />On the whole, VIVAH delivers what it promised, a well made and realistic story of two families. The movie has top-notch performances, excellent story and great music to suit the film, as well as being directed by the fabulous Sooraj R. Barjatya. It's a must see!  
    neg	  626	 Watching this Movie? l thought to myself, what a lot of garbage. These girls must have rocks for brains for even agreeing to be part of it. Waste of time watching it, faint heavens l only hired it. The acting was below standard and story was unbearable. Anyone contemplating watching this film, please save your money. The film has no credit at all. l am a real film buff and this is worse than "Attack of the Green Tomatoes".<br /><br />l only hope that this piece of trash didn't cost too much to make. Money would have been better spent on the homeless people of the world. l only hope there isn't a sequel in the pipeline.  
    pos	  2599	 A SPECIAL DAY (Ettore Scola - Italy/Canada 1977).<br /><br />Every once in a while, you come across a film that really touches a nerve. This one offers a very simple premise, almost flawlessly executed in every way and incredibly moving at the same time. It's surprising Ettore Scola's "Una giornate particulare" is relatively unheralded, even hated by some critics. Time Out calls it 'rubbish' and Leonard Maltin, somewhat milder, 'pleasant but trifling.' I disagree, not only because this film is deeply moving, but within its simple story it shows us more insights about daily life in fascist Italy than most films I've seen. The cinematography is distinctly unflashy, even a bit bland, and the storyline straightforward, which might explain the film's relative unpopularity. Considering late '70s audiences weren't exactly spoiled with great Italian films, it's even stranger this one didn't really catch on with the critics.<br /><br />The film begins with a ten-minute collage of archive footage from Hitler's visit to Italy on may 8th 1938. Set against this background, we first meet Antonietta (Loren), a lonely, love-ridden housewife with six children in a roman apartment building. One day, when her Beo escapes, she meets her neighbour Gabriele (Mastroianni), who seems to be only one in the building not attending the ceremonies. He is well-mannered, cultured and soon she is attracted to him. During the whole film, we hear the fascist rally from the radio of the concierge hollering through the courtyard. Scola playfully uses the camera to make us part of the proceedings. After the opening scene, the camera swanks across the courtyard of the modernist (hypermodern at the time) apartment block, seemingly searching for our main characters, whom we haven't met yet. <br /><br />Marcello Mastrionani and Sophia Loren are unforgettable in the two leading roles, all the more astonishing since they are cast completely against type. Canadian born John Vernon plays Loren's husband, but he is only on screen in the first and last scene. I figure his voice must have been dubbed, since he's not of Italian descent and never lived there, to my knowledge, so I cannot imagine he speaks Italian. If his voice has been dubbed, I didn't notice at all. On the contrary, he's completely believable as an Italian, even more than the rest of the cast. The story is simple but extremely effective, the performances are outstanding, the ending is just perfect and the framing doesn't come off as overly pretentious but works completely. Don't miss out on this one.<br /><br />Camera Obscura --- 9/10  
    neg	  1482	 There are some extremely talented black directors Spike Lee,Carl Franklin,Billy Dukes,Denzel and a host of others who bring well deserved credit to the film industry . Then there are the Wayans Brothers who at one time(15,years ago) had an extremely funny television show'In Living Colour' that launched the career of Jim Carrey amongst others . Now we have stupidity substituting for humour and gross out gags(toilet humour) as the standard operating procedure . People are not as stupid as those portrayed in 'Little Man' they couldn't possibly be . A baby with a full set of teeth and a tattoo is accepted as being only months old ? Baby comes with a five o'clock shadow that he shaves off . It is intimated that the baby has sex with his foster mother behind her husbands,Darryl's, back .Oh, yea that is just hilarious . As a master criminal 'Little Man' is the stupidest on planet earth . He stashes a stolen rock that is just huge in a woman's purse and then has to pursue her . Co-star Chazz Palminteri,why Chazz, offers the best line: "I'm surrounded by morons." Based, without credit, on a Chuck Jones cartoon, Baby Buggy Bunny . This is far too stupid to be even remotely funny . A clue as to how bad this film is Damon Wayans appeared on Jay Leno the other night,prior to the BAT awards and he did not,even mention this dreadful movie . When will Hollywood stop green lighting trash from the Wayans Brothers . When they get over their white mans guilt in all likelihood .  
    neg	  4380	 There is a bit of a spoiler below, which could ruin the surprise of the ONE unexpected and truly funny scene in this film. There is also information about the first film in this series.<br /><br />I caught this film on DVD, which someone gave as a gift to my roommate. It came as a set together with the first film in the "Blind Dead" series.<br /><br />This movie was certainly much worse than the first, "La Noche del Terror Ciego". In addition, many of the features of the first movie were changed significantly. To boot, the movie was dubbed in English (the first was subtitled), which I tend to find distracting.<br /><br />The concept behind the series is that in the distant past a local branch of the Knights Templar was involved in heinous and secret rituals. Upon discovery of these crimes, the local peasantry put the Templars to death in such a manner that their eyes can no longer be used, thus preventing them from returning from Hell to exact their revenge. We then jump to modern times where because of some event, the Templars arise from the dead to exact their revenge upon the villagers whose ancestors messed them up in the first place. Of course, since the undead knights have no eyes, they can only find their victims when they make some sort of noise.<br /><br />The Templars were a secretive order, from about the 12th century, coming out of the Crusades. They were only around for about 150 years, before they were suppressed in the early 1300s by the Pope and others. Because they were secretive, there were always rumors about their ceremonies, particularly for initiation. Also, because of the way the society was organized, you didn't necessarily have church officials overseeing things, which meant they didn't have an inside man when things heated up. And, because of the nature of their trials, they were tortured into confessions. The order was strongest in France, but did exist in Portugal and Spain, where the movies take place.<br /><br />Where the first movie had a virgin sacrifice and knights drinking the blood directly from the body of the virgin (breast shots here, of course, this is a horror film after all), and then, once the knights come back to life, they attack their victims by eating them alive and sucking their blood; in this sequel, this all disappears. You still have the same scene (redone, not the same footage) of them sacrificing the virgin, but they drain the blood into a bowl and drink it from that. Thus, when they come back, they just hack people up with their swords or claw people to death, which I have to say is a much less effective means of disturbing your audience. There's also a time problem: in the first film the dating is much closer to the Templars, where here they are now saying it is the 500 anniversary of the peasants burning these guys at the stake, which would date it around 1473. And the way that the Templars lose their eyes is much less interesting as well. In the first, they have them pecked out by crows. Now they are simply burned out, and in quite a ridiculous manner.<br /><br />Oh yeah, and maybe it was just me, but there seemed to be a lot of people from the first movie reappearing in this film (despite having died). Not really a problem, since the movie is completely different and not a sequel in the sense of a continuation, but odd none-the-less.<br /><br />The highlight of this movie is the rich fellow who uses a child to distract the undead while he makes a break for the jeep. The child's father had already been suckered by this rich man into making an attempt to get the jeep, so he walks out and tells her to find her father. It comes somewhat out of the blue, and is easily the funniest scene in the film. Of course, why the child doesn't die at this point is beyond me, and disappointed for horror fans.<br /><br />I couldn't possibly recommend this film to anyone. It isn't so bad that it becomes funny, so it just ends up being a mediocre horror film. The bulk of the film has several people holed up in a church, each making various attempts to go it alone in order to escape the blind dead who have them surrounded. When the film ends, you are not surprised at the outcome at all; in fact, quite disappointed. If you are into the novelty of seeing a Spanish horror film, see the first movie, which at least has some innovative ideas and not so expected outcomes.  


    PyTorchText BuketIterator

    Batch size: 10

    LABEL LENGTH TEXT
    neg	  1118	 Most college students find themselves lost in the bubble of academia, cut off from the communities in which they study and live. Their conversations are held with their fellow students and the college faculty. Steven Greenstreet's documentary is a prime example of a disillusioned college student who judges the entire community based on limited contact with a small number of its members.<br /><br />The documentary focused on a small group of individuals who were portrayed as representing large groups of the population. As is usual, the people who scream the most get the most media attention. Other than its misrepresentation of the community in which the film was set, the documentary was well made. My only dispute is that the feelings and uproar depicted in the film were attributed to the entire community rather than the few individuals who expressed them.<br /><br />Naturally it is important to examine a controversy like this and make people aware of the differences that exist between political viewpoints, but it is ridiculous to implicate an entire community of people in the actions of a few radicals.  
    neg	  1120	 Looked forward to viewing this film and seeing these great actors perform. However, I was sadly disappointed in the script and the entire plot of the story. David Duchovny,(Dr. Eugene Sands),"Connie & Carla",'04, was the doctor in the story who uses drugs and losses his license to practice medicine. Dr. Sands was visiting a night club and was able to use his medical experience to help a wounded customer and was assisted by Angelina Jolie,(Claire),"Taking Lives",'04, who immediately becomes attracted to Dr. David Sands. Timothy Hutton,(Raymond Blossom),"Kinsey",'04, plays the Big Shot Gangster and a man with all kinds of money and connections. Timothy Hutton seems to over act in most of the scenes and goes completely out of his mind trying to keep his gang members from being killed. Gary Dourdan,(Yates),"CSI-Vegas TV Series", plays a great supporting role and portrays a real COOL DUDE who is a so-called body guard for Raymond Blossom. Angelina Jolie looks beautiful and sexy with her ruby red lips which draws a great deal of attention from all the men. This film is not the greatest, but it does entertain.  
    pos	  1120	 I must say that, looking at Hamlet from the perspective of a student, Brannagh's version of Hamlet is by far the best. His dedication to stay true to the original text should be applauded. It helps the play come to life on screen, and makes it easier for people holding the text while watching, as we did while studying it, to follow and analyze the text.<br /><br />One of the things I have heard criticized many times is the casting of major Hollywood names in the play. I find that this helps viewers recognize the characters easier, as opposed to having actors that all look and sound the same that aid in the confusion normally associated with Shakespeare.<br /><br />Also, his flashbacks help to clear up many ambiguities in the text. Such as how far the relationship between Hamlet and Ophelia really went and why Fortinbras just happened to be at the castle at the end. All in all, not only does this version contain some brilliant performances by actors both familiar and not familiar with Shakespeare. It is presented in a way that one does not have to be an English Literature Ph.D to understand and enjoy it.  
    pos	  1120	 As a baseball die-hard, this movie goes contrary to what I expect in a sports movie: authentic-looking sports action, believable characters, and an original story line. While "Angels in the Outfield" fails miserably in the first category, it succeeds beautifully in the latter two. "Angels" weaves the story of Roger and J.P., two Anaheim foster kids in love with baseball but searching for a family, with that of the woebegone Angels franchise, struggling to draw fans and win games. Pushed by his deadbeat father's promise that they would be a family only when the Angels win the pennant, Roger asks for some heavenly help, and gets it in the form of diamond-dwelling spirits bent on reversing the franchise's downward spiral. And, when short-fused manager George Knox (portrayed by Danny Glover) begins believing in what Roger sees, the team suddenly has hope for turning their season around--and Roger and J.P. find something to believe in. Glover in particular gives a nice performance, and Tony Danza, playing a washed-up pitcher, also does well, despite clearly having ZERO idea of how to pitch out of the windup!  
    neg	  1121	 I have a piece of advice for the people who made this movie too, if you're gonna make a movie like this be sure you got the f/x to back it up. Also don't get a bunch of z list actors to play in it. Another thing, just about all of us have seen Jurassic Park, so don't blatantly copy it. All in all this movie sucked, f/x sucked, acting sucked, story unoriginal. Let's talk about the acting for just a second, the Carradine guy who's career peaked in 1984 when he did "Revenge of the Nerds" (which was actually a great comedy). He's not exactly z list, he can act. He just should have said no to this s--t bag. He should have did what Mark Hamill did after "Return of the Jedi" and go quietly into the night. He made his mark as a "Nerd" and that should have been that. I understand he has bills to pay, but that hardly excuses this s--t bag. Have I called this movie that yet? O.K. I just wanted to be sure. If I sound a little hostile, I apologize. I just wasted 2hrs of my life I could have spent doing something productive like watching paint peel, and I feel cheated. I'll close on that note. Thank you for your time.  
    neg	  1121	 By 1941 Columbia was a full-fledged major studio and could produce a movie with the same technical polish as MGM, Paramount or Warners. That's the best thing that could be said about "Adam Had Four Sons," a leaden soap opera with almost terminally bland performances by Ingrid Bergman (top-billed for the first time in an American film) and Warner Baxter. Bergman plays a Frenchwoman (this was the era in which Hollywood thought one foreign accent was as good as another) hired as governess to Baxter's four sons and staying on (with one interruption caused by the stock-market crash of 1907) until the boys are grown men serving in World War I. Just about everyone in the movie is so goody-good it's a relief when Susan Hayward as the villainess enters midway through — she's about the only watchable person in the movie even though she's clearly channeling Bette Davis and Vivien Leigh; it's also the first in her long succession of alcoholic roles — but the script remains saccharine and the ending is utterly preposterous. No wonder Bergman turned down the similarly plotted "The Valley of Decision" four years later.  
    neg	  1123	 I have never read the book"A wrinkle in time". To be perfectly honesty, after seeing the movie, do I really want to? Well, I shouldn't be reviewing this movie i'll start off with that. Next i'll say that the TV movie is pretty forgettable. Do you know why I say that? Because I forgot what happens in it. I told you it was forgettable. To be perfectly honest, no TV movie will ever be better than "Merlin".<br /><br />How do I describe a TV movie? I have never written a review for one before. Well, i'll just say that they usually have some celebrities. A wrinkle in time includes only one. Alfre Woodard(Or Woodward, I am not sure), the Oscar winner. <br /><br />The film has cheesy special effects, a mildly interesting plot, scenes that make you go "WTF". The movie is incredibly bad and it makes you go"WTF". What did I expect? It's a TV movie. They usually aren't good. As is this one. A wrinkle in time is a waste of time and a big time waster. To top it off, you'll most likely forget about it the second it's over. Well, maybe not the second it's over. But within a few minutes.<br /><br />A wrinkle in time:*/****  
    neg	  1123	 After watching "The Bodyguard" last night, I felt compelled to write a review of it.<br /><br />This could have been a pretty decent movie had it not been for the awful camera-work. It was beyond annoying. The angles were all wrong, it was impossible to see anything, especially during the fight sequences. The closeups were even horrible.<br /><br />The story has Sonny Chiba hiring himself out as a bodyguard to anyone willing to lead him to the top of a drug ring. He is approached by Judy Lee, who is never quite straight with Chiba. Lee's involvement in the drug ring is deeper than Chiba thought, as the Mob and another gang of thugs are after her.<br /><br />The story was decent, and despite horrible dubbing, this could have been a good movie. Given better direction and editing, I'm sure this would have been a classic Kung Foo movie. As it is, it's more like another cheesy 70's action movie.<br /><br />Note: The opening sequence has a quote familiar to "Pulp Fiction" fans, and then continues to a karate school in Times Square that is in no way related to the rest of the movie.<br /><br />Rating: 4 out of 10  
    neg	  1123	 There are some really terrific ideas in this violent movie that, if executed clearly, could have elevated it from Spaghetti-western blandness into something special. Unfortunately, A TOWN CALLED HELL is one of the worst edited movies imaginable! Scenes start and end abruptly, characters leave for long stretches, the performances (and accents) of the actors are pretty inconsistent, etc.<br /><br />Robert Shaw is a Mexican(!) revolutionary who, after taking part in wiping out a village, stays on to become a priest(!)...ten years later the village is being run by "mayor" Telly Salavas. Stella Stevens arrives looking for revenge on the man who killed her husband. Colonel Martin Landau arrives looking for Shaw. They all yell at each other A LOT and they all shoot each other A LOT. Fernando Rey is in it too (as a blind man). The performances aren't bad, but they are mightily uneven. Savalas has an accent sometimes as does Landau (who is really grating here). Shaw and Rey prove that they are incapable of really embarrassing themselves and Stevens looks pretty foxy (if a bit out of place amongst the sweaty filth).  
    neg	  1124	 The movie is plain bad. Simply awful. The string of bad movies from Bollywood has no end! They must be running out of excuses for making such awful movies (or not).<br /><br />The problem seems to be with mainly the directors. This movie has 2 good actors who have proved in the past that the have the ability to deliver great performance...but they were directed so poorly. The poor script did not help either.<br /><br />This movie has plenty of ridiculous moments and very bad editing in the first half. For instance :<br /><br />After his 1st big concert, Ajay Devgan, meets up with Om Puri (from whom he ran away some 30 years ago and talked to again) and all Om Puri finds to say is to beware of his friendship with Salman!!! What a load of crap. Seriously. Not to mention the baaad soundtrack. Whatever happened to Shankar Ehsaan Loy?<br /><br />Ajay Devgun is total miscast for portraying a rockstar.<br /><br />Only saving grace are the good performances in the second half. Ajay shines as his character shows his dark side. So does Salman as the drug addict. <br /><br />Watch it maybe only for the last half hour.  




### **Train Loop Examples**

Now let's look at a model training loop would look like. I printed the first 10 batches list of examples lengths to show how nicely they are grouped throughout the dataset!


```python
# Example of number of epochs
epochs = 1

# Example of loop through each epoch
for epoch in range(epochs):

  # Create batches - needs to be called before each loop.
  torchtext_train_dataloader.create_batches()

  # Loop through BucketIterator.
  for sample_id, batch in enumerate(torchtext_train_dataloader.batches):
    print('Batch examples lengths: %s'.ljust(20) % str([len(example['text']) for example in batch]))

    # Let's break early, you get the idea.
    if sample_id == 10:
      break
```

    Batch examples lengths: [791, 792, 792, 793, 797, 797, 799, 799, 801, 801]
    Batch examples lengths: [4823, 4832, 4859, 4895, 4944, 5025, 5150, 5309, 5313, 5450]
    Batch examples lengths: [695, 696, 696, 696, 697, 699, 699, 700, 700, 701]
    Batch examples lengths: [960, 961, 963, 963, 963, 966, 966, 967, 968, 969]
    Batch examples lengths: [1204, 1205, 1208, 1209, 1212, 1214, 1218, 1221, 1226, 1229]
    Batch examples lengths: [2639, 2651, 2651, 2672, 2692, 2704, 2707, 2712, 2720, 2724]
    Batch examples lengths: [1815, 1830, 1835, 1838, 1841, 1849, 1852, 1878, 1889, 1895]
    Batch examples lengths: [3111, 3115, 3133, 3174, 3201, 3206, 3217, 3278, 3294, 3334]
    Batch examples lengths: [3001, 3031, 3039, 3047, 3056, 3077, 3084, 3103, 3104, 3107]
    Batch examples lengths: [1053, 1053, 1056, 1057, 1060, 1067, 1073, 1077, 1078, 1080]
    Batch examples lengths: [751, 751, 756, 758, 759, 760, 761, 762, 763, 764]


## **Using PyTorchText TabularDataset**

Now I will use the TabularDataset functionality which creates the PyTorchDataset object right from our local files.

We don't need to create a custom PyTorch Dataset class to load our dataset as long as we have tabular files of our data.

### **Data to Files**

Since our dataset is scattered into multiple files, I created a function `files_to_tsv` which puts our dataset into a `.tsv` file (Tab-Separated Values).

Since I'll use the **TabularDataset** from `pytorch.data` I need to pass tabular format files.

For text data I find the Tab Separated Values format easier to deal with.

I will call the `files_to_tsv` function for each of the two partitions **train** and **test**.

The function will return the name of the `.tsv` file saved so we can use it later in PyTorchText.


```python
def files_to_tsv(partition_path, save_path='./'):
  """Parse each file in partition and keep track of sentiments.
  Create a list of pairs [tag, text]

  Arguments:

    partition_path (:obj:`str`):
      Partition used: train or test.

    save_path (:obj:`str`):
      Path where to save the final .tsv file.

  Returns:

    :obj:`str`: Filename of created .tsv file.

  """

  # List of all examples in format [tag, text].
  examples = []

  # Print partition.
  print(partition_path)

  # Loop through each sentiment.
  for sentiment in ['pos', 'neg']:

    # Find path for sentiment.
    sentiment_path = os.path.join(partition_path, sentiment)

    # Get all files from path sentiment.
    files_names = os.listdir(sentiment_path)

    # For each file in path sentiment.
    for file_name in tqdm(files_names, desc=f'{sentiment} Files'):

      # Get file content.
      file_content = io.open(os.path.join(sentiment_path, file_name), mode='r', encoding='utf-8').read()

      # Fix any format errors.
      file_content = fix_text(file_content)

      # Append sentiment and file content.
      examples.append([sentiment, file_content])

  # Create a TSV file with same format `sentiment  text`.
  examples = ["%s\t%s"%(example[0], example[1]) for example in examples]

  # Create file name.
  tsv_filename = os.path.basename(partition_path) + '_pos_neg_%d.tsv'%len(examples)

  # Write to TSV file.
  io.open(os.path.join(save_path, tsv_filename), mode='w', encoding='utf-8').write('\n'.join(examples))

  # Return TSV file name.
  return tsv_filename


# Path where to save tsv file.
data_path = '/content'

# Convert train files to tsv file.
train_filename = files_to_tsv(partition_path='/content/aclImdb/train', save_path=data_path)

# Convert test files to tsv file.
test_filename = files_to_tsv(partition_path='/content/aclImdb/test', save_path=data_path)
```

    /content/aclImdb/train
    pos Files: 100% |████████████████████████████████| 12500/12500 [00:34<00:00, 367.26it/s]
    neg Files: 100% |████████████████████████████████| 12500/12500 [00:21<00:00, 573.00it/s]

    /content/aclImdb/test
    pos Files: 100% |████████████████████████████████| 12500/12500 [00:11<00:00, 1075.80it/s]
    neg Files: 100% |████████████████████████████████| 12500/12500 [00:12<00:00, 1037.94it/s]





### **TabularDataset**

Here I setup the data fields for PyTorchText. We have to tell the library how to handle each column of the `.tsv` file. For this we need to create `data.Field` objects for each column.

`text_tokenizer`:
For this example I don't use an actual tokenizer for the `text` column but I need to create one because it requires as input. I created a dummy tokenizer that returns same value. Depending on the project, here is where you will have your own tokenizer. It needs to take as input text and output a list.

`label_tokenizer`
The label tokenizer is also a dummy tokenizer. This is where you will have a encoder to transform labels to ids.

Since we have two `.tsv` files it's great that we can use the `.split` function from **TabularDataset** to handle two files at the same time one for train and the other one for test.

Find more details about **torchtext.data** functionality [here](https://torchtext.readthedocs.io/en/latest/data.html#dataset-batch-and-example).


```python
# Text tokenizer function - dummy tokenizer to return same text.
# Here you will use your own tokenizer.
text_tokenizer = lambda x : x

# Label tokenizer - dummy label encoder that returns same label.
# Here you will add your own label encoder.
label_tokenizer = lambda x: x

# Data field for text column - invoke tokenizer.
TEXT = torchtext.data.Field(sequential=True, tokenize=text_tokenizer, lower=False)

# Data field for labels - invoke tokenize label encoder.
LABEL = torchtext.data.Field(sequential=True, tokenize=label_tokenizer, use_vocab=False)

# Create data fields as tuples of description variable and data field.
datafields = [("label", LABEL),
              ("text", TEXT)]

# Since we have have tab separated data we use TabularDataset
train_dataset, valid_dataset = torchtext.data.TabularDataset.splits(

                                                # Path to train and validation.
                                                path=data_path,

                                                # Train data filename.
                                                train=train_filename,

                                                # Validation file name.
                                                validation=test_filename,

                                                # Format of local files.
                                                format='tsv',

                                                # Check if we have header.
                                                skip_header=False,

                                                # How to handle fields.
                                                fields=datafields)
```

### **PyTorchText Bucket Iterator Dataloader**

I'm using same setup as in the **PyTorchText Bucket Iterator Dataloader** code cell section. The only difference is in the `sort_key` since there is different way to access example attributes (we had dictionary format before).


```python
# Group similar length text sequences together in batches.
torchtext_train_dataloader, torchtext_valid_dataloader = torchtext.data.BucketIterator.splits(

                              # Datasets for iterator to draw data from
                              (train_dataset, valid_dataset),

                              # Tuple of train and validation batch sizes.
                              batch_sizes=(train_batch_size, valid_batch_size),

                              # Device to load batches on.
                              device=device,

                              # Function to use for sorting examples.
                              sort_key=lambda x: len(x.text),


                              # Repeat the iterator for multiple epochs.
                              repeat=True,

                              # Sort all examples in data using `sort_key`.
                              sort=False,

                              # Shuffle data on each epoch run.
                              shuffle=True,

                              # Use `sort_key` to sort examples in each batch.
                              sort_within_batch=True,
                              )

# Print number of batches in each split.
print('Created `torchtext_train_dataloader` with %d batches!'%len(torchtext_train_dataloader))
print('Created `torchtext_valid_dataloader` with %d batches!'%len(torchtext_valid_dataloader))
```

    Created `torchtext_train_dataloader` with 2500 batches!
    Created `torchtext_valid_dataloader` with 1250 batches!


### **Compare DataLoaders**

Let's compare the PyTorch DataLoader batches with the PyTorchText BucketIterator batches created with TabularDataset. We can see how nicely examples of similar length are grouped in same batch with PyTorchText.

**Note:** *When using the PyTorchText BucketIterator, make sure to call `create_batches()` before looping through each batch! Else you won't get any output form the iterator.*


```python
# Loop through regular dataloader.
print('PyTorch DataLoader\n')
for batch in torch_train_dataloader:

  # Let's check batch size.
  print('Batch size: %d\n'% len(batch['text']))
  print('LABEL\tLENGTH\tTEXT'.ljust(10))

  # Print each example.
  for text, label in zip(batch['text'], batch['label']):
    print('%s\t%d\t%s'.ljust(10) % (label, len(text), text))
  print('\n')

  # Only look at first batch. Reuse this code in training models.
  break


# Create batches - needs to be called before each loop.
torchtext_train_dataloader.create_batches()

# Loop through BucketIterator.
print('PyTorchText BuketIterator\n')
for batch in torchtext_train_dataloader.batches:

  # Let's check batch size.
  print('Batch size: %d\n'% len(batch))
  print('LABEL\tLENGTH\tTEXT'.ljust(10))

  # Print each example.
  for example in batch:
    print('%s\t%d\t%s'.ljust(10) % (example.label, len(example.text), example.text))
  print('\n')

  # Only look at first batch. Reuse this code in training models.
  break
```

    PyTorch DataLoader

    Batch size: 10

    LABEL LENGTH TEXT
    pos	  1742	 As a child I preferred the first Care Bear movie since this one seemed so dark. I always sat down and watched the first one. As I got older I learned to prefer this one. What I do think is that this film is too dark for infants, but as you get older you learn to treasure it since you understand it more, it doesn't seem as dark as it was back when you were a child.<br /><br />This movie, in my opinion, is better than the first one, everything is so much deeper. It may contradict the first movie but you must ignore the first movie to watch this one. The cubs are just too adorable, I rewind that 'Flying My Colors' scene. I tend to annoy everyone by singing it.<br /><br />The sound track is great! A big hand to Carol and Dean Parks. I love every song in this movie, I have downloaded them all and is all I am listening to, I'm listening to 'Our beginning' also known as 'Recalling' at the moment. I have always preferred this sound track to the first one, although I just totally love Carol Kings song in the first movie 'Care-A-Lot'.<br /><br />I think the animation is great, the animation in both movies are fantastic. I was surprised when I sat down and watched it about 10 years later and saw that the animation for the time was excellent. It was really surprising.<br /><br />There is not a lot of back up from other people to say that this movie is great, but it is. I do not think it is weird/strange. I think it is a wonderful movie.<br /><br />Basically, this movie is about how the Care Bears came about and to defeat the Demon, Dark Heart. The end is surprising and again, beats any 'Pokemon Movie' with the Care Bears Moral issues. It leaves an effect on you. Again this movie can teach everyone at all ages about morality.  
    pos	  1475	 Worry not, Disney fans--this special edition DVD of the beloved Cinderella won't turn into a pumpkin at the strike of midnight. One of the most enduring animated films of all time, the Disney-fide adaptation of the gory Brothers Grimm fairy tale became a classic in its own right, thanks to some memorable tunes (including "A Dream Is a Wish Your Heart Makes," "Bibbidi-Bobbidi-Boo," and the title song) and some endearingly cute comic relief. The famous slipper (click for larger image) We all know the story--the wicked stepmother and stepsisters simply won't have it, this uppity Cinderella thinking she's going to a ball designed to find the handsome prince an appropriate sweetheart, but perseverance, animal buddies, and a well-timed entrance by a fairy godmother make sure things turn out all right. There are a few striking sequences of pure animation--for example, Cinderella is reflected in bubbles drifting through the air--and the design is rich and evocative throughout. It's a simple story padded here agreeably with comic business, particularly Cinderella's rodent pals (dressed up conspicuously like the dwarf sidekicks of another famous Disney heroine) and their misadventures with a wretched cat named Lucifer. There's also much harrumphing and exposition spouting by the King and the Grand Duke. It's a much simpler and more graceful work than the more frenetically paced animated films of today, which makes it simultaneously quaint and highly gratifying.  
    pos	  1279	 Seldom do I ever encounter a film so completely fulfilling that I must speak about it immediately. This movie is definitely some of the finest entertainment available and it is highly authentic. I happened to see the dubbed version but I'm on my way right now to grab the DVD remaster with original Chinese dialogue. Still, the dubbing didn't get in the way and sometimes provided some seriously funny humour: "Poison Clan rocks the world!!!"<br /><br />The story-telling stays true to Chinese methods of intrigue, suspense, and inter-personal relationships. You can expect twists and turns as the identities of the 5 venoms are revealed and an expert pace.<br /><br />The martial arts fight choreography is in a class of its own and must be seen to be believed. It's like watching real animals fight each other, but construed from their own arcane martial arts forms. Such level of skill amongst the cast is unsurpassed in modern day cinema.<br /><br />The combination provides for a serious dose of old Chinese culture and I recommend it solely on the basis of the film's genuine intent to tell a martial arts story and the mastery of its execution. ...Of course, if you just want to see people pummel each other, along with crude forms of ancient Chinese torture, be my guest!  
    pos	  1071	 I'm sure that most people already know the story-the miserly Ebenezer Scrooge gets a visit from three spirits (the Ghosts of Christmas Past, Present and Yet to Come) who highlight parts of his life in the hopes of saving his soul and changing his ways. Dickens' classic story in one form or another has stood the test of time to become a beloved holiday favorite.<br /><br />While I grew up watching the 1951 version starring Alastair Sims, and I believe that he is the definitive Scrooge, I have been impressed with this version, which was released when I was in high school. George C. Scott plays a convincing and mean Ebenezer Scrooge, and the actors playing the ghosts are rather frightening and menacing. David Warner is a good Bob Cratchit as well.<br /><br />This version is beautifully filmed, and uses more modern filming styles (for the 1980's) which make it more palatable for my children than the 1951 black and white version.<br /><br />This is a worthy adaptation of the story and is one that I watch almost every year at some point in the Christmas season.  
    neg	  876	 What was an exciting and fairly original series by Fox has degraded down to meandering tripe. During the first season, Dark Angel was on my weekly "must see" list, and not just because of Jessica Alba.<br /><br />Unfortunately, the powers-that-be over at Fox decided that they needed to "fine-tune" the plotline. Within 3 episodes of the season opener, they had totally lost me as a viewer (not even to see Jessica Alba!). I found the new characters that were added in the second season to be too ridiculous and amateurish. The new plotlines were stretching the continuity and credibility of the show too thin. On one of the second season episodes, they even had Max sleeping and dreaming - where the first season stated she biologically couldn't sleep.<br /><br />The moral of the story (the one that Hollywood never gets): If it works, don't screw with it!<br /><br />azjazz  
    pos	  1981	 Greta Garbo's American film debut is an analogy of how our lives can be swept off course by fate and our actions, as in a torrent, causing us to lose a part of ourselves along the way.<br /><br />Greta plays Leonora, a poor peasant girl in love with Ricardo Cortez's character Don Rafael, a landowner. Ricardo is in love with her too, but is too easily influenced by his domineering mother. Leonora ends up homeless and travels to Paris, where she becomes a famous opera singer and develops the reputation for being a loose woman. In reality, part of her attitude is bitterness over Rafael's abandonment.<br /><br />She returns to her home to visit her family and eventually confronts Rafael. Surprisingly, no one knows that she's the famous La Brunna, and Garbo acts up her role as the diva she truly was and re prised with such cool haughtiness in her later portrayals.<br /><br />Ricardo Cortez reminds one a lot of Valentino in looks in this part, and he was groomed to be a Valentino clone by MGM, though he never thought he could be in reality and he was right. He is believable in an unsympathetic part as a weak willed Mama's boy, and allows himself to age realistically but comically at the end of the movie. He fails to win Leonora when she returns home, and later when he follows her, his courage is undermined.<br /><br />This movie is beautifully shot, with brilliant storm sequences and the sets depicting Spain at the time are authentic looking. There are also some fine secondary performances by old timers Lucien Littlefield, Tully Marshall, and Mack Swain.<br /><br />Although this is a story of lost love and missed chances, I don't think Leonora and Rafael would have been happy together, as he needed a more traditional wife and she was very much a career woman, and I don't think would have been happy in a small village. The ending is true to life and pulls no punches.<br /><br />See this one as Garbo's American film debut and a precursor of things to come  
    pos	  1007	 *What I Like About SPOILERS* Teenager Holly Tyler (Amanda Bynes) goes to live with older sister Valerie (Jennie Garth) to avoid moving to Japan with her father; but she doesn't know the half of the wacky things that will happen to her from now on, and not only to her, but to her sister, her friends Gary (Wesley Jonathan) and Tina (Alison Munn), boyfriend Henry (Michael McMillian), crush Vince (Nick Zano), Valerie's boyfriend Jeff (Simon Rex), first boss (then firefighter then husband) Vic (Dan Cortese), annoying colleague Lauren (Leslie Grossman) and second boss Peter (?) If you don't have a funny bone in your body, please skip this; if you like only veeeery sophisticated comedy this isn't for you; if you like a funny, sometimes touching show with two hot chicks who can act in the lead (and none other than the fabulous 'Mary Cherry' from Popular - Leslie Grossman - in the main cast), then what the hell are you waiting for? You're welcome to Casa De Tyler! What I Like About You (2002-2006): 8.  
    pos	  318	 This movie is wonderful. The writing, directing, acting all are fantastic. Very witty and clever script. Quality performances by actors, Ally Sheedy is strong and dynamic and delightfully quirky. Really original and heart-warmingly unpredicatable. The scenes are alive with fresh energy and really talented production.  
    pos	  1846	 In Le Million, Rene Clair, one of the cinema's great directors and great pioneers, created a gem of light comedy which for all its lightness is a groundbreaking and technically brilliant film which clearly influenced subsequent film-makers such as the Marx Brothers, Lubitsch, and Mamoulian. The plot, a witty story of a poor artist who wins a huge lottery jackpot but has to search frantically all over town for the missing ticket, is basically just a device to support a series of wonderfully witty comic scenes enacted in a dream world of the director's imagination.<br /><br />One of the most impressive things about this film is that, though it is set in the middle of Paris and includes nothing actually impossible, it achieves a sustained and involving fairy-tale/fantasy atmosphere, in which it seems quite natural that people sing as much as they talk, or that a tussle over a stolen jacket should take on the form of a football game. Another memorable element is that Le Million includes what may be the funniest opera ever put on film (O that blonde-braided soprano! "I laugh, ha! ha!") Also a delight is the casting: Clair has assembled a group of amazing, sharply different character actors, each of them illustrating with deadly satiric accuracy a bourgeois French "type," so that the film seems like a set of Daumier prints come to life.<br /><br />The hilarity takes a little while to get rolling, and I found the characters not as emotionally engaging as they can be even in a light comedy (as they are, for instance, in many Lubitsch films.) For these reasons I refrained from giving it the highest rating. But these minor cavils shouldn't distract from an enthusiastic recommendation.<br /><br />Should you see it? By all means. Highly recommended whether you want a classic and influential work of cinema or just a fun comedy.  
    pos	  1260	 Before I comment about this movie, you should realize that when I saw this movie, I expected the typical crap, horror, B-movie and just wanted to have fun. Jack Frost is one that not only delivers but is actually one of the best that I've seen in a long time. Scott McDonald is great as Jack Frost, in fact I think he has a future in being psychopaths in big time movies if ever given the chance. McDonald is a serial killer who becomes a snowman through some stupid accidental mix of ridiculous elements. As soon as that snowman starts moving around and killing people, though, you will find it hard not to laugh. The lines that are said are completely retarded but really funny. The fact that the rest of the cast completely over-acts just adds to stupidity of the film, but it's stupidity is it's genius. The scene where the snowman is with the teenage girl is truly classic in B-movie, horror film fashion. I truly hope there is a sequel and I'll be right there to watch it on whatever cable channel does it. Of course it's only fun to watch the first few times and it's not exactly a good work of motion picture technology, but I just like to see snowmen kill people. I gave it a 7 out of 10, this is a great movie for dates and couples in the late hours.  


    PyTorchText BuketIterator

    Batch size: 10

    LABEL LENGTH TEXT
    neg	  1118	 Most college students find themselves lost in the bubble of academia, cut off from the communities in which they study and live. Their conversations are held with their fellow students and the college faculty. Steven Greenstreet's documentary is a prime example of a disillusioned college student who judges the entire community based on limited contact with a small number of its members.<br /><br />The documentary focused on a small group of individuals who were portrayed as representing large groups of the population. As is usual, the people who scream the most get the most media attention. Other than its misrepresentation of the community in which the film was set, the documentary was well made. My only dispute is that the feelings and uproar depicted in the film were attributed to the entire community rather than the few individuals who expressed them.<br /><br />Naturally it is important to examine a controversy like this and make people aware of the differences that exist between political viewpoints, but it is ridiculous to implicate an entire community of people in the actions of a few radicals.  
    neg	  1120	 Looked forward to viewing this film and seeing these great actors perform. However, I was sadly disappointed in the script and the entire plot of the story. David Duchovny,(Dr. Eugene Sands),"Connie & Carla",'04, was the doctor in the story who uses drugs and losses his license to practice medicine. Dr. Sands was visiting a night club and was able to use his medical experience to help a wounded customer and was assisted by Angelina Jolie,(Claire),"Taking Lives",'04, who immediately becomes attracted to Dr. David Sands. Timothy Hutton,(Raymond Blossom),"Kinsey",'04, plays the Big Shot Gangster and a man with all kinds of money and connections. Timothy Hutton seems to over act in most of the scenes and goes completely out of his mind trying to keep his gang members from being killed. Gary Dourdan,(Yates),"CSI-Vegas TV Series", plays a great supporting role and portrays a real COOL DUDE who is a so-called body guard for Raymond Blossom. Angelina Jolie looks beautiful and sexy with her ruby red lips which draws a great deal of attention from all the men. This film is not the greatest, but it does entertain.  
    pos	  1120	 I must say that, looking at Hamlet from the perspective of a student, Brannagh's version of Hamlet is by far the best. His dedication to stay true to the original text should be applauded. It helps the play come to life on screen, and makes it easier for people holding the text while watching, as we did while studying it, to follow and analyze the text.<br /><br />One of the things I have heard criticized many times is the casting of major Hollywood names in the play. I find that this helps viewers recognize the characters easier, as opposed to having actors that all look and sound the same that aid in the confusion normally associated with Shakespeare.<br /><br />Also, his flashbacks help to clear up many ambiguities in the text. Such as how far the relationship between Hamlet and Ophelia really went and why Fortinbras just happened to be at the castle at the end. All in all, not only does this version contain some brilliant performances by actors both familiar and not familiar with Shakespeare. It is presented in a way that one does not have to be an English Literature Ph.D to understand and enjoy it.  
    pos	  1120	 As a baseball die-hard, this movie goes contrary to what I expect in a sports movie: authentic-looking sports action, believable characters, and an original story line. While "Angels in the Outfield" fails miserably in the first category, it succeeds beautifully in the latter two. "Angels" weaves the story of Roger and J.P., two Anaheim foster kids in love with baseball but searching for a family, with that of the woebegone Angels franchise, struggling to draw fans and win games. Pushed by his deadbeat father's promise that they would be a family only when the Angels win the pennant, Roger asks for some heavenly help, and gets it in the form of diamond-dwelling spirits bent on reversing the franchise's downward spiral. And, when short-fused manager George Knox (portrayed by Danny Glover) begins believing in what Roger sees, the team suddenly has hope for turning their season around--and Roger and J.P. find something to believe in. Glover in particular gives a nice performance, and Tony Danza, playing a washed-up pitcher, also does well, despite clearly having ZERO idea of how to pitch out of the windup!  
    neg	  1121	 I have a piece of advice for the people who made this movie too, if you're gonna make a movie like this be sure you got the f/x to back it up. Also don't get a bunch of z list actors to play in it. Another thing, just about all of us have seen Jurassic Park, so don't blatantly copy it. All in all this movie sucked, f/x sucked, acting sucked, story unoriginal. Let's talk about the acting for just a second, the Carradine guy who's career peaked in 1984 when he did "Revenge of the Nerds" (which was actually a great comedy). He's not exactly z list, he can act. He just should have said no to this s--t bag. He should have did what Mark Hamill did after "Return of the Jedi" and go quietly into the night. He made his mark as a "Nerd" and that should have been that. I understand he has bills to pay, but that hardly excuses this s--t bag. Have I called this movie that yet? O.K. I just wanted to be sure. If I sound a little hostile, I apologize. I just wasted 2hrs of my life I could have spent doing something productive like watching paint peel, and I feel cheated. I'll close on that note. Thank you for your time.  
    neg	  1121	 By 1941 Columbia was a full-fledged major studio and could produce a movie with the same technical polish as MGM, Paramount or Warners. That's the best thing that could be said about "Adam Had Four Sons," a leaden soap opera with almost terminally bland performances by Ingrid Bergman (top-billed for the first time in an American film) and Warner Baxter. Bergman plays a Frenchwoman (this was the era in which Hollywood thought one foreign accent was as good as another) hired as governess to Baxter's four sons and staying on (with one interruption caused by the stock-market crash of 1907) until the boys are grown men serving in World War I. Just about everyone in the movie is so goody-good it's a relief when Susan Hayward as the villainess enters midway through — she's about the only watchable person in the movie even though she's clearly channeling Bette Davis and Vivien Leigh; it's also the first in her long succession of alcoholic roles — but the script remains saccharine and the ending is utterly preposterous. No wonder Bergman turned down the similarly plotted "The Valley of Decision" four years later.  
    neg	  1123	 I have never read the book"A wrinkle in time". To be perfectly honesty, after seeing the movie, do I really want to? Well, I shouldn't be reviewing this movie i'll start off with that. Next i'll say that the TV movie is pretty forgettable. Do you know why I say that? Because I forgot what happens in it. I told you it was forgettable. To be perfectly honest, no TV movie will ever be better than "Merlin".<br /><br />How do I describe a TV movie? I have never written a review for one before. Well, i'll just say that they usually have some celebrities. A wrinkle in time includes only one. Alfre Woodard(Or Woodward, I am not sure), the Oscar winner. <br /><br />The film has cheesy special effects, a mildly interesting plot, scenes that make you go "WTF". The movie is incredibly bad and it makes you go"WTF". What did I expect? It's a TV movie. They usually aren't good. As is this one. A wrinkle in time is a waste of time and a big time waster. To top it off, you'll most likely forget about it the second it's over. Well, maybe not the second it's over. But within a few minutes.<br /><br />A wrinkle in time:*/****  
    neg	  1123	 After watching "The Bodyguard" last night, I felt compelled to write a review of it.<br /><br />This could have been a pretty decent movie had it not been for the awful camera-work. It was beyond annoying. The angles were all wrong, it was impossible to see anything, especially during the fight sequences. The closeups were even horrible.<br /><br />The story has Sonny Chiba hiring himself out as a bodyguard to anyone willing to lead him to the top of a drug ring. He is approached by Judy Lee, who is never quite straight with Chiba. Lee's involvement in the drug ring is deeper than Chiba thought, as the Mob and another gang of thugs are after her.<br /><br />The story was decent, and despite horrible dubbing, this could have been a good movie. Given better direction and editing, I'm sure this would have been a classic Kung Foo movie. As it is, it's more like another cheesy 70's action movie.<br /><br />Note: The opening sequence has a quote familiar to "Pulp Fiction" fans, and then continues to a karate school in Times Square that is in no way related to the rest of the movie.<br /><br />Rating: 4 out of 10  
    neg	  1123	 There are some really terrific ideas in this violent movie that, if executed clearly, could have elevated it from Spaghetti-western blandness into something special. Unfortunately, A TOWN CALLED HELL is one of the worst edited movies imaginable! Scenes start and end abruptly, characters leave for long stretches, the performances (and accents) of the actors are pretty inconsistent, etc.<br /><br />Robert Shaw is a Mexican(!) revolutionary who, after taking part in wiping out a village, stays on to become a priest(!)...ten years later the village is being run by "mayor" Telly Salavas. Stella Stevens arrives looking for revenge on the man who killed her husband. Colonel Martin Landau arrives looking for Shaw. They all yell at each other A LOT and they all shoot each other A LOT. Fernando Rey is in it too (as a blind man). The performances aren't bad, but they are mightily uneven. Savalas has an accent sometimes as does Landau (who is really grating here). Shaw and Rey prove that they are incapable of really embarrassing themselves and Stevens looks pretty foxy (if a bit out of place amongst the sweaty filth).  
    neg	  1124	 The movie is plain bad. Simply awful. The string of bad movies from Bollywood has no end! They must be running out of excuses for making such awful movies (or not).<br /><br />The problem seems to be with mainly the directors. This movie has 2 good actors who have proved in the past that the have the ability to deliver great performance...but they were directed so poorly. The poor script did not help either.<br /><br />This movie has plenty of ridiculous moments and very bad editing in the first half. For instance :<br /><br />After his 1st big concert, Ajay Devgan, meets up with Om Puri (from whom he ran away some 30 years ago and talked to again) and all Om Puri finds to say is to beware of his friendship with Salman!!! What a load of crap. Seriously. Not to mention the baaad soundtrack. Whatever happened to Shankar Ehsaan Loy?<br /><br />Ajay Devgun is total miscast for portraying a rockstar.<br /><br />Only saving grace are the good performances in the second half. Ajay shines as his character shows his dark side. So does Salman as the drug addict. <br /><br />Watch it maybe only for the last half hour.  




### **Train Loop Examples**

Now let's look at a model training loop would look like. I printed the first 10 batches list of examples lengths to show how nicely they are grouped throughout the dataset!

We see that we get same exact behavior as we did when using PyTorch Dataset. Now it depends on which way is easier for you to use PyTorchText BucketIterator: with PyTorch Dataset or with PyTorchText TabularDataset


```python
# Example of number of epochs.
epochs = 1

# Example of loop through each epoch.
for epoch in range(epochs):

  # Create batches - needs to be called before each loop.
  torchtext_train_dataloader.create_batches()

  # Loop through BucketIterator.
  for sample_id, batch in enumerate(torchtext_train_dataloader.batches):
    # Put all example.text of batch in single array.
    batch_text = [example.text for example in batch]

    print('Batch examples lengths: %s'.ljust(20) % str([len(text) for text in batch_text]))

    # Let's break early, you get the idea.
    if sample_id == 10:
      break
```

    Batch examples lengths: [791, 791, 792, 792, 793, 797, 797, 799, 799, 801]
    Batch examples lengths: [4766, 4823, 4832, 4859, 4895, 4944, 5025, 5150, 5309, 5313]
    Batch examples lengths: [695, 695, 696, 696, 696, 697, 699, 699, 699, 700]
    Batch examples lengths: [958, 959, 960, 961, 963, 963, 963, 966, 966, 967]
    Batch examples lengths: [1200, 1203, 1204, 1205, 1208, 1209, 1212, 1214, 1218, 1221]
    Batch examples lengths: [2621, 2628, 2639, 2651, 2651, 2672, 2690, 2704, 2705, 2712]
    Batch examples lengths: [1811, 1812, 1815, 1830, 1835, 1838, 1841, 1849, 1852, 1878]
    Batch examples lengths: [3104, 3107, 3111, 3115, 3133, 3174, 3201, 3206, 3217, 3278]
    Batch examples lengths: [3000, 3001, 3001, 3031, 3039, 3047, 3056, 3075, 3084, 3103]
    Batch examples lengths: [1046, 1050, 1053, 1053, 1054, 1057, 1060, 1067, 1073, 1077]
    Batch examples lengths: [749, 751, 751, 756, 758, 759, 760, 761, 762, 763]


## **Final Note**

If you made it this far **Congrats!** 🎊 and **Thank you!** 🙏 for your interest in my tutorial!

I've been using this code for a while now and I feel it got to a point where is nicely documented and easy to follow.

Of course is easy for me to follow because I built it. That is why any feedback is welcome and it helps me improve my future tutorials!

If you see something wrong please let me know by opening an issue on my [ml_things GitHub repository](https://github.com/gmihaila/ml_things/issues)!

A lot of tutorials out there are mostly a one-time thing and are not being maintained. I plan on keeping my tutorials up to date as much as I can.

## **Contact** 🎣

🦊 GitHub: [gmihaila](https://github.com/gmihaila)

🌐 Website: [gmihaila.github.io](https://gmihaila.github.io/)

👔 LinkedIn: [mihailageorge](https://medium.com/r/?url=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fmihailageorge)

📬 Email: [georgemihaila@my.unt.edu.com](mailto:georgemihaila@my.unt.edu.com?subject=GitHub%20Website)
