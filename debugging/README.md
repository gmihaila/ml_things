# Python Debugging tool

To set debugging:

```python
import pdb; pdb.set_trace()
```

For example:

```python
def add(a, b):
    '''
    This is the test:
    >>> add(2, 2)
    5
    '''
    import pdb; pdb.set_trace()
    return a + b
```
