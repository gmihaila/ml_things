# Perform Unit Test in Python

* Make sure to keep the unit testing code in a separate file with name `test_` in front of the `.py` script.
* When writing code that needs testing make sure you raise any `TypeError` or `ValueError` or other types of errors associated with the fucntions input values (if any).

## Example using `.py` script

### Code script

`subtract.py` is a python script which contains a function called `subtract()` which takes 2 arguments and returns the subtracted value:

```python
from math import pi

def subtract(a, b):
    if (type(a) not in [int, float]) and (type(b) not in [int, float]):
        raise TypeError("Function values don't match the accepted types int/float.")
        
    if b == 0:
        raise ValueError("The denominator cannot be zero.")
        
    return a/b
```

### Unit Test script

`test_subtract.py` will contain all code needed to perform unit test on our `subtract()` function from `subtract.py`:

```python
import unittest
from subtract import subtract
from math import pi


class TestSubtract(unittest.TestCase):
    def test_subtract(self):
        # Test a few values
        self.assertAlmostEqual(subtract(1, 2), 1/2)
        self.assertAlmostEqual(subtract(-3, 5), -3/5)
        self.assertAlmostEqual(subtract(13, 45), 13/45)
        
    def test_values(self):
        # Make sure value errors are raised when necessary
        self.assertRaises(ValueError, subtract, 1, 0)
        
    def test_type(self):
        # Make sure type erors are raised when necessary
        self.assertRaises(TypeError, subtract, (1,True))
        self.assertRaises(TypeError, subtract, (False, True))
        self.assertRaises(TypeError, subtract, ("one","two"))
```

### Run unittest

In order to run python unittest for this example:

`python -m unittest test_subtract.py`

Will need to return:

```bash
...
----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK
```

The `OK` tells us everything went well! The `3 tests` are represented by the number of testing fucntions written in `test_subtract.py`


## Example using `.ipynb` jupyter notebook

### A notebook cell with a function:

```python
def add(a, b):
    return a + b
```

### A notebook cell (the last one in the notebook) that contains a test case. The last line in the cell runs the test case when the cell is executed:

```python
import unittest

class TestNotebook(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(2, 2), 5)


unittest.main(argv=[''], verbosity=2, exit=False)
```

### Output:

```bash
test_add (__main__.TestNotebook) ... FAIL

======================================================================
FAIL: test_add (__main__.TestNotebook)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "<ipython-input-15-4409ad9ffaea>", line 6, in test_add
    self.assertEqual(add(2, 2), 5)
AssertionError: 4 != 5

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (failures=1)
```



