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
"""Functions related to web applications"""

import os
import requests


def download_from(url, path):
    """Download file from url.

    :param url: web path of file.
    :param path: path to save the file.
    :return: path where file was saved
    """

    # get file name from url
    file_name = os.path.basename(url)
    # create directory frm path if it doesn't exist
    os.makedirs(path) if not os.path.isdir(path) else None
    # find path where file will be saved
    file_path = os.path.join(path, file_name)
    # check if file already exists
    if not os.path.isfile(file_path):
        # if files does not exist - download
        response = requests.get(url)
        file_size = open(file_path, 'wb').write(response.content)
        file_size = '{0:.2f}MB'.format(file_size / 1024)
        # print file details
        print("%s %s" % (file_path, file_size))
    return file_path
