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
"""Functions that are used on array like variables/objects"""


def pad_array(variable_length_array, fixed_length=None, axis=1, pad_value=0.0):
    """Pad variable length array to a fixed numpy array.
    It can handle single arrays [1,2,3] or nested arrays [[1,2],[3]].

    :param
      variable_length_array: Single arrays [1,2,3] or nested arrays [[1,2],[3]].
    :param
      fixed_length: max length of rows for numpy.
    :param
      axis: directions along rows: 1 or columns: 0
    :param
      pad_value: what value to use as padding, default is 0.
    :return:
      numpy_array:  axis=1: fixed numpy array shape [len of array, fixed_length].
                    axis=0: fixed numpy array shape [fixed_length, len of array].
    """

    # padded array in numpy format
    numpy_array = None

    if axis not in [1, 0]:
        # axis value is wrong
        raise ValueError("`axis` value needs to be 1 for row padding \
                    or 0 for column padding!")

    # find fixed_length if no value given
    fixed_length = max([len(row) for row in variable_length_array]) if fixed_length is None else fixed_length

    # array of arrays
    if isinstance(variable_length_array[0], list) or isinstance(
            variable_length_array[0], np.ndarray):

        if axis == 1:
            # perform padding on rows
            numpy_array = np.ones((len(variable_length_array), fixed_length)) * pad_value
            # verify each row
            for numpy_row, array_row in zip(numpy_array, variable_length_array):
                # concatenate array row if it is longer
                array_row = array_row[:fixed_length]
                numpy_row[:len(array_row)] = array_row

        elif axis == 0:
            # make sure all rows have same length
            if not all([len(row) == len(variable_length_array[0])
                        for row in variable_length_array]):
                raise ValueError("`variable_length_array` need to have same row length for column padding `axis=0`!")
            # padding on columns
            if fixed_length >= len(variable_length_array):
                # need to pad
                numpy_array = np.ones((fixed_length, len(variable_length_array[0]))) * pad_value
                numpy_array[:len(variable_length_array)] = variable_length_array
            else:
                # need to cut array
                numpy_array = np.array(variable_length_array[:fixed_length])

        return numpy_array

    # array of values
    elif isinstance(variable_length_array, list) or isinstance(
            variable_length_array, np.ndarray):

        if axis == 1:
            # perform padding on rows
            numpy_array = np.ones(fixed_length) * pad_value
            variable_length_array = variable_length_array[:fixed_length]
            numpy_array[:len(variable_length_array)] = variable_length_array

        elif axis == 0:
            # padding on columns
            numpy_array = np.ones((fixed_length, len(variable_length_array))) * pad_value
            numpy_array[0] = variable_length_array

        return numpy_array

    else:
        # array is not a valid format
        raise ValueError("`variable_length_array` is not a valid format.")


def batch_array(list_values, batch_size):
    """Split a list into batches/chunks. Last batch size is remaining of list values.

    :param list_values: can be any kind of list/array.
    :param batch_size: int value of the batch length.
    :return: List of batches from list_values.

    Note:
      This is also called chunking. I call it batches since I use it more in ML.
    """

    if isinstance(list_values, list) or isinstance(list_values, np.ndarray):
        # make sure to warn user if `list_value` has correct type

        if len(list_values) < batch_size:
            # make sure batch size is not greater than length of list
            warnings.warn("`batch_size` is greater than length of `list_values`!")

            return [list_values]

        # create new list of batches
        batched_list = [list_values[i * batch_size:(i + 1) * batch_size] for i in
                        range((len(list_values) + batch_size - 1) // batch_size)]

        return batched_list

    else:
        # raise error if `list_values` is not of type array
        raise ValueError("`list_values` must be of type list!")
