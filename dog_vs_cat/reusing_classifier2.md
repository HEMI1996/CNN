

```python
from keras.models import load_model
new_classifier = load_model('classifier2.h5')
```

    C:\Users\Hemanth\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
import numpy as np
from keras.preprocessing import image
def prediction(path):
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = new_classifier.predict(test_image)
    if result[0][0] == 1:
        pred = 'dog'
    else:
        pred = 'cat'
    print(pred)
```


```python
path = 'dataset/new_prediction/cat_or_dog_1.jpeg'
prediction(path)
```

    cat
    


```python
path = 'dataset/new_prediction/cat_or_dog_2.jpeg'
prediction(path)
```

    dog
    


```python
path = 'dataset/new_prediction/cat1.jpeg'
prediction(path)
```

    dog
    


```python
path = 'dataset/new_prediction/dog1.jpeg'
prediction(path)
```

    dog
    


```python
path = 'dataset/new_prediction/cat2.jpeg'
prediction(path)
```

    dog
    


```python
path = 'dataset/new_prediction/dog2.jpeg'
prediction(path)
```

    dog
    


```python
path = 'dataset/new_prediction/cat3.jpeg'
prediction(path)
```

    dog
    


```python
path = 'dataset/new_prediction/dog3.jpeg'
prediction(path)
```

    dog
    


```python
path = 'dataset/new_prediction/cat4.jpeg'
prediction(path)
```

    cat
    


```python
path = 'dataset/new_prediction/dog4.jpeg'
prediction(path)
```

    dog
    


```python
## 3 wrong predictions
```
