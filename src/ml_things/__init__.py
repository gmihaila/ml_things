# coding=utf-8
# Copyright 2020 George Mihaila.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__version__ = "0.0.1"

# make sure warnings are imported
import warnings
# always show deprecation warnings
warnings.simplefilter('always', DeprecationWarning)

# functions imports
from .array_functions import (pad_array,
                              batch_array,
                              )

from .web_related import (download_from,
                          )

from .plot_functions import (plot_array,
                             plot_dict,
                             plot_confusion_matrix,
                             )

# text type function
from .text_functions import (clean_text,
                             )

# installed ftfy to fix any UNICODE problems in text data
from ftfy import fix_text

# alternative names
from .array_functions import batch_array as chunk_array
