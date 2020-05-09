## Sentiments: Positive & Negative

## Imports

```python
import io
import os
import requests
```

## Function

```python
def positive_negative_words(path=".sentiments"):
    """Download positive and negative files and return list of positive and negative words.
    Args:
        path: Temporary path to save sentiment files

    Returns:
        positive_words: List of positive words.
        negative_words: List of negative words.
    """
    # files download urls
    positive_url = "https://raw.githubusercontent.com/gmihaila/machine_learning_things/master/sentiments/positive-words.txt"
    negative_url = "https://raw.githubusercontent.com/gmihaila/machine_learning_things/master/sentiments/negative-words.txt"
    # build path
    os.makedirs(path) if not os.path.isdir(path) else None
    # create file paths
    positive_path = os.path.join(path, "positive-words.txt")
    negative_path = os.path.join(path, "negative-words.txt")
    # download files
    open(positive_path, 'wb').write(requests.get(positive_url).content) if not os.path.isfile(positive_path) else None
    open(negative_path, 'wb').write(requests.get(negative_url).content) if not os.path.isfile(negative_path) else None
    # read file
    positive_words = io.open(positive_path, encoding='UTF-8').read().strip().split('\n')
    negative_words = io.open(negative_path, encoding='UTF-8').read().strip().split('\n')

    return positive_words, negative_words
```

## Use Function

```python
positive_words, negative_words = positive_negative_words()
```
<br>

## Details:
* Positive: `2,006` words.
* Negative: `4,783` words

* Files are borrowed from [shekhargulati/sentiment-analysis-python](https://github.com/shekhargulati/sentiment-analysis-python)

* These files contains a list of POSITIVE and NEGATIVE opinion words (or sentiment words).

* These files and the papers can all be downloaded from 
     http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
 
* If you use this list, please cite one of the following two papers:
 
    * Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
        Proceedings of the ACM SIGKDD International Conference on Knowledge 
        Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, 
        Washington, USA, 
    * Bing Liu, Minqing Hu and Junsheng Cheng. "Opinion Observer: Analyzing 
        and Comparing Opinions on the Web." Proceedings of the 14th 
        International World Wide Web conference (WWW-2005), May 10-14, 
        2005, Chiba, Japan.
 
## Notes:
  * The appearance of an opinion word in a sentence does not necessarily  
    mean that the sentence expresses a positive or negative opinion. 
    See the paper below:

    Bing Liu. "Sentiment Analysis and Subjectivity." An chapter in 
       Handbook of Natural Language Processing, Second Edition, 
       (editors: N. Indurkhya and F. J. Damerau), 2010.

  * You will notice many misspelled words in the list. They are not 
    mistakes. They are included as these misspelled words appear 
    frequently in social media content. 

