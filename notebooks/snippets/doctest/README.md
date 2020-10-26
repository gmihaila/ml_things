# Perform test on functions similar to unittest

Doctest is a standard testing tools

## Doctest inside jupyter notebook

### A notebook cell with a function and a test case in a docstring:

```python
def add(a, b):
    '''
    This is a test:
    >>> add(2, 2)
    5
    '''
    return a + b
```

### A notebook cell (the last one in the notebook) that runs all test cases in the docstrings:

```python
import doctest
doctest.testmod(verbose=True)
```
### Output:

```shell
Trying:
    add(2, 2)
Expecting:
    5
**********************************************************************
File "__main__", line 4, in __main__.add
Failed example:
    add(2, 2)
Expected:
    5
Got:
    4
1 items had no tests:
    __main__
**********************************************************************
1 items had failures:
   1 of   1 in __main__.add
1 tests in 2 items.
0 passed and 1 failed.
***Test Failed*** 1 failures.
```
