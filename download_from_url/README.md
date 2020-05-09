# Download file from url inside of Python

## Imports

```python
import os
import requests
```

## Function:

```python
def download_from(url, path):
  file_name = os.path.basename(url)

  os.makedirs(path) if not os.path.isdir(path) else None

  file_path = os.path.join(path, file_name)

  if not os.path.isfile(file_path):
    response = requests.get(url)
    file_size = open(file_path, 'wb').write(response.content)
    file_size = '{0:.2f}MB'.format(file_size/1024)
    print("%s %s" % (file_path, file_size))
  return file_path
```

## Use Function:
```python
download_from(url="https://raw.githubusercontent.com/gmihaila/machine_learning_things/master/sentiments/positive-words.txt", path="senwst")
```
