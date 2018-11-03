
# Part  1 - Building the CNN


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

# Part 2 - Fitting the CNN to the images


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
    250/250 [==============================] - 269s 1s/step - loss: 0.6928 - acc: 0.5962 - val_loss: 0.6473 - val_acc: 0.6280
    Epoch 2/25
    250/250 [==============================] - 94s 374ms/step - loss: 0.5944 - acc: 0.6798 - val_loss: 0.6205 - val_acc: 0.6735
    Epoch 3/25
    250/250 [==============================] - 93s 372ms/step - loss: 0.5681 - acc: 0.7054 - val_loss: 0.5518 - val_acc: 0.7200
    Epoch 4/25
    250/250 [==============================] - 93s 372ms/step - loss: 0.5476 - acc: 0.7216 - val_loss: 0.5722 - val_acc: 0.6970
    Epoch 5/25
    250/250 [==============================] - 93s 374ms/step - loss: 0.5281 - acc: 0.7359 - val_loss: 0.5159 - val_acc: 0.7540
    Epoch 6/25
    250/250 [==============================] - 93s 373ms/step - loss: 0.5109 - acc: 0.7491 - val_loss: 0.5525 - val_acc: 0.7310
    Epoch 7/25
    250/250 [==============================] - 93s 370ms/step - loss: 0.4929 - acc: 0.7566 - val_loss: 0.5316 - val_acc: 0.7430
    Epoch 8/25
    250/250 [==============================] - 92s 368ms/step - loss: 0.4789 - acc: 0.7650 - val_loss: 0.5300 - val_acc: 0.7600
    Epoch 9/25
    250/250 [==============================] - 92s 367ms/step - loss: 0.4677 - acc: 0.7707 - val_loss: 0.5600 - val_acc: 0.7335
    Epoch 10/25
    250/250 [==============================] - 91s 366ms/step - loss: 0.4545 - acc: 0.7809 - val_loss: 0.5216 - val_acc: 0.7515
    Epoch 11/25
    250/250 [==============================] - 91s 366ms/step - loss: 0.4435 - acc: 0.7935 - val_loss: 0.5058 - val_acc: 0.7670
    Epoch 12/25
    250/250 [==============================] - 92s 367ms/step - loss: 0.4242 - acc: 0.8015 - val_loss: 0.5370 - val_acc: 0.7545
    Epoch 13/25
    250/250 [==============================] - 92s 367ms/step - loss: 0.4130 - acc: 0.8054 - val_loss: 0.5707 - val_acc: 0.7345
    Epoch 14/25
    250/250 [==============================] - 92s 367ms/step - loss: 0.4019 - acc: 0.8158 - val_loss: 0.5510 - val_acc: 0.7460
    Epoch 15/25
    250/250 [==============================] - 93s 374ms/step - loss: 0.3940 - acc: 0.8228 - val_loss: 0.5367 - val_acc: 0.7600
    Epoch 16/25
    250/250 [==============================] - 94s 374ms/step - loss: 0.3776 - acc: 0.8281 - val_loss: 0.5597 - val_acc: 0.7560
    Epoch 17/25
    250/250 [==============================] - 94s 376ms/step - loss: 0.3584 - acc: 0.8405 - val_loss: 0.5632 - val_acc: 0.7370
    Epoch 18/25
    250/250 [==============================] - 187s 747ms/step - loss: 0.3524 - acc: 0.8449 - val_loss: 0.5785 - val_acc: 0.7625
    Epoch 19/25
    250/250 [==============================] - 139s 557ms/step - loss: 0.3416 - acc: 0.8495 - val_loss: 0.5798 - val_acc: 0.7620
    Epoch 20/25
    250/250 [==============================] - 94s 374ms/step - loss: 0.3260 - acc: 0.8580 - val_loss: 0.6940 - val_acc: 0.7395
    Epoch 21/25
    250/250 [==============================] - 94s 376ms/step - loss: 0.3062 - acc: 0.8649 - val_loss: 0.6074 - val_acc: 0.7605
    Epoch 22/25
    250/250 [==============================] - 95s 379ms/step - loss: 0.3007 - acc: 0.8734 - val_loss: 0.6700 - val_acc: 0.7545
    Epoch 23/25
    250/250 [==============================] - 94s 374ms/step - loss: 0.2879 - acc: 0.8804 - val_loss: 0.6549 - val_acc: 0.7495
    Epoch 24/25
    250/250 [==============================] - 95s 380ms/step - loss: 0.2692 - acc: 0.8879 - val_loss: 0.7165 - val_acc: 0.7540
    Epoch 25/25
    250/250 [==============================] - 94s 377ms/step - loss: 0.2731 - acc: 0.8827 - val_loss: 0.6240 - val_acc: 0.7630
    




    <keras.callbacks.History at 0x1df783d1630>




```python
classifier.save('classifier1.h5')
```


```python
training_set.class_indices
```




    {'cats': 0, 'dogs': 1}



1. In first line we load image and we will the target size equal to target size given when training is     performed.
2. Then we convert the image to an array. Then test image will be 3-dimensional (64, 64, 3)
3. Now we need to one more dimension... i.e., batch size. we add batch size at index zero, because          prediction method expects batch size at index zero .. so we include axis = 0 


```python
# Making new predictions
from keras.preprocessing import image
```


```python
import numpy as np
test_image = image.load_img('dataset/new_prediction/cat_or_dog_1.jpeg', target_size = (64, 64))
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
    


```python
test_image = image.load_img('dataset/new_prediction/cat_or_dog_2.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
```

    dog
    


```python
test_image = image.load_img('dataset/new_prediction/cat_or_dog_3.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
```

    dog
    


```python
def prediction(path):
    test_image = image.load_img(path, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    return prediction
```


```python
path = 'dataset/new_prediction/cat3.jpeg'
pred = prediction(path)
print(pred)
```

    dog
    


```python
path = 'dataset/new_prediction/cat1.jpeg'
pred = prediction(path)
print(pred)
```

    dog
    


```python
path = 'dataset/new_prediction/cat2.jpeg'
pred = prediction(path)
print(pred)
```

    dog
    


```python
path = 'dataset/new_prediction/cat4.jpeg'
pred = prediction(path)
print(pred)
```

    dog
    


```python
path = 'dataset/new_prediction/cat_or_dog_1.jpeg'
pred = prediction(path)
print(pred)
```

    cat
    
