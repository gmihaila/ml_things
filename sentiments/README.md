## Opinion Lexicon: Positive & Negative

### Python Snippet:

```python
Import io

positive_words = io.open("/path/to/positive-words.txt", encoding='UTF-8').read().strip().split('\n')
negative_words = io.open("/path/to/negative-words.txt", encoding='UTF-8').read().strip().split('\n')
```

### This is borrowed from [shekhargulati
/
sentiment-analysis-python](https://github.com/shekhargulati/sentiment-analysis-python)

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
 
#### Notes:
  * The appearance of an opinion word in a sentence does not necessarily  
    mean that the sentence expresses a positive or negative opinion. 
    See the paper below:

    Bing Liu. "Sentiment Analysis and Subjectivity." An chapter in 
       Handbook of Natural Language Processing, Second Edition, 
       (editors: N. Indurkhya and F. J. Damerau), 2010.

  * You will notice many misspelled words in the list. They are not 
    mistakes. They are included as these misspelled words appear 
    frequently in social media content. 

