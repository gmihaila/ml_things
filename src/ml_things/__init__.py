# flake8: noqa
# coding=utf-8
# Copyright 2020 George Mihaila.

__version__ = "0.0.1"

import warnings

# functions imports
from .array_functions import (pad_array,
                              batch_array)
from .web_related import (download_from)
from .plot_functions import (plot_array,
                             plot_dict,
                             plot_confusion_matrix)
from .text_functions import (clean_text)
# installed ftfy to fix any UNICODE problems in text data
from ftfy import fix_text

# alternative names
from .array_functions import batch_array as chunk_array
