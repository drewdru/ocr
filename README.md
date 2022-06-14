# ocr
Text recognition from an images

# Create Dataset
create dataset with structure:
```
    dataset
    |
    ...train
    ...|
    ......classA
    ......|
    .........imgA
    .........imgB
    ............
    .........imgZ
    ......classB
    ...
    ...validation
    ...|
    ......classA
    ......classB
    ...
```
Set CLASSES in ./main.py
Change `Dense` with class count in ./train.py
near `model = Sequential([`
