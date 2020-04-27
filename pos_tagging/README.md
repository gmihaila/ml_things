# POS Tagging

## Python Code Snippet

### Imports:

```python
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
``

### Function:

```python
def word_pos(word):
  """Convert word to POS TAG if exists in WordNet.
  Imports: 'from nltk.corpus import wordnet as wn'.
  Download WordNet: 'nltk.download('wordnet')'
  Pos Names: 'NOUN', 'VERB', 'ADJECTIVE', 'ADJECTIVE SATELITE', 'ADVERB'
  Pos Tags: 'n', 'v', 'a', 's', 'r'

  Note:
      Regarding ADJECTIVE SATELITE:
      Certain adjectives bind minimal meaning. e.g. "dry", "good", &tc. 
      Each of these is the center of an adjective synset in WN.
      Adjective satellites imposes additional commitments on top of the meaning
      of the central adjective, e.g. "arid" = "dry" + a particular context 
      (i.e. climates).
  """
  # from nltk.corpus import wordnet as wn
  # nltk.download('wordnet')
  # Certain adjectives bind minimal meaning. e.g. "dry", "good", &tc. Each of these is the center of an adjective synset in WN.
  tag = wn.synsets(word)
  if len(tag) > 0:
    tag = tag[0].pos()
    # https://wordnet.princeton.edu/documentation/wndb5wn
    pos_decode = {'n':'NOUN', 'v':'VERB', 'a':'ADJECTIVE', 's':'ADJECTIVE SATELITE', 'r':'ADVERB'}
    return pos_decode[tag]
  else:
    return None
```
