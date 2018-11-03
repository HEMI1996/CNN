

```python
# Importing the Keras libraries and Packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
```

    C:\Users\Hemanth\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
# Initializing the CNN
classifier = Sequential()
```


```python
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
```


```python
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))
```


```python
# Adding one more Convolution and Pooling layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
```


```python
# Step 3 - Flattening
classifier.add(Flatten())
```


```python
# Step 4 - Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
```


```python
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```


```python
from keras.preprocessing.image import ImageDataGenerator
```


```python
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)
```


```python
test_datagen = ImageDataGenerator(rescale = 1./255)
```


```python
training_set = train_datagen.flow_from_directory('dataset/training_set', 
                                                 target_size = (64, 64),
                                                 batch_size = 32, 
                                                 class_mode = 'binary')
```

    Found 8000 images belonging to 2 classes.
    


```python
test_set = test_datagen.flow_from_directory('dataset/test_set', 
                                            target_size = (64, 64),
                                            batch_size = 25, 
                                            class_mode = 'binary')
```

    Found 2000 images belonging to 2 classes.
    


```python
classifier.fit_generator(training_set, 
                         steps_per_epoch = 250,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 80)
```

    Epoch 1/25
    250/250 [==============================] - 306s 1s/step - loss: 0.6391 - acc: 0.6330 - val_loss: 0.5730 - val_acc: 0.7140
    Epoch 2/25
    250/250 [==============================] - 121s 483ms/step - loss: 0.5742 - acc: 0.7007 - val_loss: 0.5424 - val_acc: 0.7305
    Epoch 3/25
    250/250 [==============================] - 115s 458ms/step - loss: 0.5287 - acc: 0.7330 - val_loss: 0.5269 - val_acc: 0.7440
    Epoch 4/25
    250/250 [==============================] - 115s 459ms/step - loss: 0.5020 - acc: 0.7509 - val_loss: 0.4899 - val_acc: 0.7660
    Epoch 5/25
    250/250 [==============================] - 113s 454ms/step - loss: 0.4907 - acc: 0.7596 - val_loss: 0.5030 - val_acc: 0.7605
    Epoch 6/25
    250/250 [==============================] - 113s 453ms/step - loss: 0.4612 - acc: 0.7823 - val_loss: 0.5053 - val_acc: 0.7630
    Epoch 7/25
    250/250 [==============================] - 113s 452ms/step - loss: 0.4443 - acc: 0.7848 - val_loss: 0.4645 - val_acc: 0.7825
    Epoch 8/25
    250/250 [==============================] - 113s 453ms/step - loss: 0.4325 - acc: 0.7943 - val_loss: 0.4812 - val_acc: 0.7855
    Epoch 9/25
    250/250 [==============================] - 113s 453ms/step - loss: 0.4221 - acc: 0.8004 - val_loss: 0.4521 - val_acc: 0.7840
    Epoch 10/25
    250/250 [==============================] - 114s 455ms/step - loss: 0.4000 - acc: 0.8139 - val_loss: 0.4798 - val_acc: 0.7815
    Epoch 11/25
    250/250 [==============================] - 114s 454ms/step - loss: 0.3848 - acc: 0.8253 - val_loss: 0.4579 - val_acc: 0.8015
    Epoch 12/25
    250/250 [==============================] - 113s 454ms/step - loss: 0.3706 - acc: 0.8256 - val_loss: 0.4455 - val_acc: 0.8090
    Epoch 13/25
    250/250 [==============================] - 113s 454ms/step - loss: 0.3588 - acc: 0.8375 - val_loss: 0.4580 - val_acc: 0.8025
    Epoch 14/25
    250/250 [==============================] - 114s 454ms/step - loss: 0.3528 - acc: 0.8470 - val_loss: 0.4808 - val_acc: 0.8035
    Epoch 15/25
    250/250 [==============================] - 113s 452ms/step - loss: 0.3381 - acc: 0.8494 - val_loss: 0.4636 - val_acc: 0.7960
    Epoch 16/25
    250/250 [==============================] - 113s 453ms/step - loss: 0.3271 - acc: 0.8564 - val_loss: 0.4395 - val_acc: 0.8050
    Epoch 17/25
    250/250 [==============================] - 114s 454ms/step - loss: 0.3104 - acc: 0.8640 - val_loss: 0.4273 - val_acc: 0.8195
    Epoch 18/25
    250/250 [==============================] - 113s 454ms/step - loss: 0.2997 - acc: 0.8665 - val_loss: 0.4988 - val_acc: 0.7905
    Epoch 19/25
    250/250 [==============================] - 113s 452ms/step - loss: 0.2945 - acc: 0.8729 - val_loss: 0.4656 - val_acc: 0.8165
    Epoch 20/25
    250/250 [==============================] - 114s 455ms/step - loss: 0.2759 - acc: 0.8812 - val_loss: 0.4807 - val_acc: 0.8125
    Epoch 21/25
    250/250 [==============================] - 113s 453ms/step - loss: 0.2642 - acc: 0.8861 - val_loss: 0.4628 - val_acc: 0.8125
    Epoch 22/25
    250/250 [==============================] - 114s 455ms/step - loss: 0.2522 - acc: 0.8964 - val_loss: 0.4846 - val_acc: 0.8075
    Epoch 23/25
    250/250 [==============================] - 113s 453ms/step - loss: 0.2352 - acc: 0.9017 - val_loss: 0.5343 - val_acc: 0.8025
    Epoch 24/25
    250/250 [==============================] - 116s 464ms/step - loss: 0.2281 - acc: 0.9027 - val_loss: 0.5292 - val_acc: 0.8115
    Epoch 25/25
    250/250 [==============================] - 113s 454ms/step - loss: 0.2166 - acc: 0.9127 - val_loss: 0.4875 - val_acc: 0.8180
    




    <keras.callbacks.History at 0x17e49e00eb8>




```python
classifier.save('classifier2.h5')
```


```python
training_set.class_indices
```




    {'cats': 0, 'dogs': 1}




```python
from keras.preprocessing import image
import numpy as np
test_image = image.load_img('dataset/new_prediction/cat4.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
```

    cat
    
