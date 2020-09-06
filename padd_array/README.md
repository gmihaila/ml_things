# Padding variable length array

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

* **What this is?**
  Function designed to padd a variable length array (list or list of lists format) to a numpy evenly shaped.

* **When to use?**
  When creating data structures to train model that require batching and consistent shapes when dealing with uneven length data example (i.e. text sequences with variable length).



## Function

```python
def pad_numpy(variable_length_array, fixed_length=None, axis=1):
    """Pad variable length array to a fixed numpy array.
    It can handle single arrays [1,2,3] or nested arrays [[1,2],[3]].
    Args:
        variable_length_array: Single arrays [1,2,3] or nested arrays [[1,2],[3]].
        fixed_length: max length of rows for numpy.
        axis: directions along rows: 1 or columns: 0

    Returns:
        numpy_array: axis=1: fixed numpy array shape [len of array, fixed_length].
                     axis=0: fixed numpy array shape [fixed_length, len of array].

    Unittest using `doctest`:
    >>> pad_numpy([1,2,3], 2, 0)
    array([[1., 2., 3.],
           [0., 0., 0.]])
    >>> pad_numpy([[1,2],[3,4]], 3, 0)
    array([[1., 2.],
           [3., 4.],
           [0., 0.]])
    >>> pad_numpy([1,2,3], 2, 1)
    array([1., 2.])
    >>> pad_numpy([[1,2],[3,4]], 3, 1)
    array([[1., 2., 0.],
           [3., 4., 0.]])
    """

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
        # perform pading on rows
        numpy_array = np.zeros((len(variable_length_array), fixed_length))
        # verify each row
        for numpy_row, array_row in zip(numpy_array, variable_length_array):
          # concatenate array row if it is longer
          array_row = array_row[:fixed_length]
          numpy_row[:len(array_row)] = array_row

      elif axis == 0:
        # make sure all rows have same length
        if not all([len(row)==len(variable_length_array[0]) \
                    for row in variable_length_array]):
          raise ValueError("`variable_length_array` need to have same row length for column padding `axis=0`!")
        # padding on columns
        numpy_array = np.zeros((fixed_length, len(variable_length_array[0])))
        numpy_array[:len(variable_length_array)] = variable_length_array

      return numpy_array

    # array of values
    elif isinstance(variable_length_array, list) or isinstance(
        variable_length_array, np.ndarray):

      if axis == 1:
        # perform pading on rows
        numpy_array = np.zeros(fixed_length)
        variable_length_array = variable_length_array[:fixed_length]
        numpy_array[:len(variable_length_array)] = variable_length_array

      elif axis == 0:
        # padding on columns
        numpy_array = np.zeros((fixed_length, len(variable_length_array)))
        numpy_array[0] = variable_length_array

      return numpy_array
    
    else:
      # array is not a valid format
      raise ValueError("`variable_length_array` is not a valid format.")
```

## Unittest

Perform unittest using `doctest` inside jupyter notebook:

```python
import doctest
doctest.testmod(verbose=True)
```

Output:

```
Trying:
    pad_numpy([1,2,3], 2, 0)
Expecting:
    array([[1., 2., 3.],
           [0., 0., 0.]])
ok
Trying:
    pad_numpy([[1,2],[3,4]], 3, 0)
Expecting:
    array([[1., 2.],
           [3., 4.],
           [0., 0.]])
ok
Trying:
    pad_numpy([1,2,3], 2, 1)
Expecting:
    array([1., 2.])
ok
Trying:
    pad_numpy([[1,2],[3,4]], 3, 1)
Expecting:
    array([[1., 2., 0.],
           [3., 4., 0.]])
ok
1 items had no tests:
    __main__
1 items passed all tests:
   4 tests in __main__.pad_numpy
4 tests in 2 items.
4 passed and 0 failed.
Test passed.
TestResults(failed=0, attempted=4)
```
&copy; George Mihaila, 2020
