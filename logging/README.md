# My Cusotm Logger File Function

## Imports:

```python
import logging
```

## Custom logger function:

```python
def custom_logger(file_log, filemode='a', date_format="%m-%d-%Y_%H:%M:%S"):
    """Create custom logger to be used across module files.
    Args:
        file_log: Name of file log.
        filemode: Either 'w' to write new file [overwrite old log] or 'a' to append to recent log.
        date_format: Date format for logg. Default is: '%m-%d-%Y_%H:%M:%S'

    Returns:
        logger object
    """
    # logger format
    logger_print_format = "%(asctime)s - %(levelname)s - %(filename)s/%(funcName)s: %(message)s"
    # define file handler and set formatter
    file_handler = logging.FileHandler(file_log)
    file_handler.setFormatter(logging.Formatter(logger_print_format, datefmt=date_format))
    # setup logger config
    logging.basicConfig(format=logger_print_format, datefmt=date_format, level=logging.INFO, filemode=filemode)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    # return created logger
    return logger
```

## Example:

```python
logger = custom_logger(file_log="file.log", filemode='w')
```

## Snippets:

### common examples
```python
logger.info("my info")
logger.warning("this is a warning")

logger.debug('A debug message')
logger.info('An info message')
logger.warning('Something is not right.')
logger.error('A Major error has happened.')
logger.critical('Fatal error. Cannot continue')
```


### exceptions log

```python
try:
  1 / 0
except ZeroDivisionError:
  logger.exception("Division by zero problem")
return
```

```python
try:
    assert 1 == 0 
except AssertionError as err:
    logger.exception("My assert failed :( ")
    raise err
```

