# Text Analysis

Currently working with dialogue `.tsv` files that contain `context` and `response` columns.

## Imports

```python
import io
import os
import keras
import nltk
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
```

## Function

```python
def text_analysis_dialogues(path, save_tokenizer=False, lowercase=False,
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', use_top=50,
                            font_size=22, plot_width=50, plot_height=8, plot_dpi=100,
                            label_rotation=45, sentiment_path=".sentiments"):
    """Text analysis statistics: number of tokens, length of sequences, different types of word frequencies.
    If you decide to save tokenizer and want to load it:
    with open(tokenizer_json) as f:
        data = json.load(f)
        tokenizer = keras.preprocessing.text.tokenizer_from_json(data)
    Args:
        path: Path of '.tsv' file or files directory.
        save_tokenizer: Save tokenizer or not.
        lowercase:
        filters:
        use_top:
        font_size:
        plot_width:
        plot_height:
        plot_dpi:
        label_rotation:
        sentiment_path:

    Returns:
        None
    """
    # set plot style
    plt.style.use("seaborn-dark")
    # variables
    n_dialogues = 0
    n_tokens_dialogues = []  # n_tokens_context + n_tokens_response
    n_tokens_contexts = []
    n_tokens_responses = []
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=None, filters=filters,
                                                   lower=lowercase, split=' ')
    # deal with stop words
    nltk.download('wordnet')
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # deal with sentiment words
    sentiment_path = ".sentiments"
    # files download urls
    positive_url = "https://raw.githubusercontent.com/gmihaila/machine_learning_things/master/sentiments/positive-words.txt"
    negative_url = "https://raw.githubusercontent.com/gmihaila/machine_learning_things/master/sentiments/negative-words.txt"
    # build path
    os.makedirs(sentiment_path) if not os.path.isdir(sentiment_path) else None
    # create file paths
    positive_path = os.path.join(sentiment_path, "positive-words.txt")
    negative_path = os.path.join(sentiment_path, "negative-words.txt")
    # download files
    open(positive_path, 'wb').write(requests.get(positive_url).content) if not os.path.isfile(positive_path) else None
    open(negative_path, 'wb').write(requests.get(negative_url).content) if not os.path.isfile(negative_path) else None
    # read file
    positive_words = io.open(positive_path, encoding='UTF-8').read().strip().split('\n')
    negative_words = io.open(negative_path, encoding='UTF-8').read().strip().split('\n')

    # check if path is folder
    if os.path.isdir(path):
        # get all file names in path
        files = os.listdir(path)
        image_path = os.path.join(path, "plot_%s.png" % os.path.basename(path))
        tokenizer_json_path = os.path.join(path, "tokenizer_%s.json" % os.path.basename(path))

    # check if path is file
    elif os.path.isfile(path):
        file_name = path.split('/')[-1]
        files = [file_name]
        file_id = os.path.basename(path)
        file_id = file_id.split(".")[0]
        output_path = os.path.dirname(path)
        image_path = os.path.join(output_path, "plot_%s.png" % file_id)
        tokenizer_json_path = os.path.join(output_path, "tokenizer_%s.json" % file_id)

    else:
        raise ValueError('ERROR: Invalid path or file.')

    print("Parsing each file.")
    for file_name in tqdm(files):
        if len(files) == 1:
            file_path = path
        else:
            file_path = os.path.join(path, file_name)

        # read tsv file
        try:
            data_frame = pd.read_csv(filepath_or_buffer=file_path, sep='\t')
            # get values
            inputs = data_frame['context'].astype(str).values
            outputs = data_frame['response'].astype(str).values
            # fit tokenizer
            tokenizer.fit_on_texts(inputs + outputs)
            # number of instances
            n_dialogues += len(inputs)
            # count tokens
            n_tokens_contexts += [len(context.split()) for context in inputs]
            n_tokens_responses += [len(response.split()) for response in outputs]

        except Exception as e:
            print("ERROR: File/s need to be '.tsv'.")
            print("ERROR: ", e, "\nTRACE:", traceback.format_exc())

    n_tokens_dialogues += [n_con + n_resp for n_con, n_resp in zip(n_tokens_contexts, n_tokens_responses)]

    tokens_details = {'Number of Context-Dialogue Pairs': len(n_tokens_dialogues),
                      'Number of Tokens': sum(n_tokens_dialogues),
                      'Vocabulary Size': len(tokenizer.word_counts),
                      'Average Tokens per Context-Dialogue Pair': np.average(n_tokens_dialogues),
                      'Average Tokens per Context': np.average(n_tokens_contexts),
                      'Average Tokens per Response': np.average(n_tokens_responses)
                      }
    print("Finished 'tokens_details'.")

    # save tokenizer
    if save_tokenizer is True:
        print("Saving tokenizer to %s" % tokenizer_json_path)
        tokenizer_json = tokenizer.to_json()
        with io.open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    # pos tagging
    print("POS Tagging.")
    pos_decode = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJECTIVE', 's': 'ADJECTIVE SATELITE', 'r': 'ADVERB'}
    pos_counts = {name: 0 for name in pos_decode.values()}
    for word, count in tokenizer.word_counts.items():
        pos_tag = nltk.corpus.wordnet.synsets(word)
        if len(pos_tag) > 0:
            pos_tag = pos_tag[0].pos()
            pos_name = pos_decode[pos_tag]
            pos_counts[pos_name] += 1

    # all words
    print("All words to DataFrame.")
    words_counts_df = pd.DataFrame(list(tokenizer.word_counts.items()))
    words_counts_df.columns = ["word", "count"]
    words_counts_df = words_counts_df.sort_values('count', ascending=False).reset_index(drop=True)

    # specific words
    print("Filter 'only_stop_words_pd'.")
    only_stop_words_pd = words_counts_df[[True if word in stop_words else False for word in
                                          words_counts_df["word"]]][:use_top]
    print("Filter 'without_stop_words_pd'.")
    without_stop_words_pd = words_counts_df[[False if word in stop_words else True for word in
                                             words_counts_df["word"]]][:use_top]

    print("Filter 'using_positive_words_pd'.")
    using_positive_words_pd = words_counts_df[[True if word in positive_words else False for word in
                                               words_counts_df["word"]]][:use_top]
    print("Filter 'using_negative_words_pd'.")
    using_negative_words_pd = words_counts_df[[True if word in negative_words else False for word in
                                               words_counts_df["word"]]][:use_top]

    words_counts_df = words_counts_df[:use_top]

    pos_counts_df = pd.DataFrame(list(pos_counts.items()))
    pos_counts_df.columns = ["word", "count"]
    pos_counts_df = pos_counts_df.sort_values('count', ascending=False).reset_index(drop=True)

    # plots
    print("Plotting.")
    plt.rcParams.update({'font.size': font_size})

    plots_dict = {"Data Tokens Details": tokens_details,
                  "All Words": words_counts_df,
                  "Without Stop Words": without_stop_words_pd,
                  "Only Stop Words": only_stop_words_pd,
                  "Only Positive Words": using_positive_words_pd,
                  "Only Negative Words": using_negative_words_pd,
                  "POS Tags Counts": pos_counts_df}

    fig = plt.figure(figsize=(plot_width, plot_height * len(plots_dict) - 1))
    fig.subplots_adjust(hspace=1)
    fig.suptitle(t='Text Analysis Top %s Tokens' % use_top,
                 fontweight='bold',
                 fontsize=70)

    # loop for each plot
    for plot_index, (plot_name, plot_df) in enumerate(plots_dict.items()):
        plt.subplot(len(plots_dict), 1, (plot_index + 1))

        if plot_name == "Data Tokens Details":
            # plot token details
            text = "\n".join(["%s : %s" % (name, f"{int(value):,d}") for name, value in plot_df.items()])
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='w', alpha=0.5)
            plt.axis('off')
            # place a text box in upper left in axes coords
            plt.text(0, 0, text, fontsize=2.5 * font_size,
                     horizontalalignment='left',
                     verticalalignment='bottom', bbox=props)
        else:
            # plot box plots
            x_values = plot_df['word'].values
            y_values = plot_df['count'].values
            plt.title(plot_name)
            plt.bar(x=x_values, height=y_values)
            plt.ylabel('Count')
            plt.xticks(rotation=label_rotation)
            [plt.text(index - 0.1, value, f"{int(value):,d}", rotation=label_rotation) for index, value in
             enumerate(y_values)];
            plt.grid()
    # save figure to image
    print("Saving figure to %s" % image_path)
    fig.savefig(fname=image_path, dpi=plot_dpi, bbox_inches='tight', pad_inches=0.5)
    return
```

## Use Function

```python
text_analysis_dialogues(path="/path/to/file.tsv")
```

## Note:
* Currently working with dialogue `.tsv` files that contain `context` and `response` columns.
