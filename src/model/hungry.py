from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import keras.backend as kb

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Model
from keras.optimizers import SGD, RMSprop, Adam

import matplotlib.pyplot as plt

from tensorflow.keras import mixed_precision
import h5py

main_path = '../../dataset/raw/Food-101/images/'
saved_path_first = '../../models/first.3.{epoch:02d}-{val_loss:.2f}.keras'
saved_path_second = '../../models/second.3.{epoch:02d}-{val_loss:.2f}.keras'

img_size = (299, 299)
batch_size = 8

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("Loading metadata...")

class_to_index = {}
index_to_class = {}

with open('../../dataset/raw/Food-101/meta/classes.txt', 'r') as file:
    classes = [line.strip() for line in file.readlines()]
    class_to_index = dict(zip(classes, range(len(classes))))
    index_to_class = dict(zip(range(len(classes)), classes))
    class_to_index = {v: k for k, v in index_to_class.items()}

print(classes)

print("Setting up Image Generator...")
data_generator = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 45,
    width_shift_range = 0.125,
    height_shift_range = 0.125,
    horizontal_flip = True,
    vertical_flip = True,
    rescale = 1./255,
    fill_mode = 'nearest'
)

print("Loading training data into the Data Frame..")
train_df = pd.read_csv('../../dataset/raw/Food-101/meta/verified_train.tsv', sep='\t', header=None, names=['path', 'label'])
train_df = train_df[train_df['label'] == 1]  # Use only verified images
train_df['label'] = train_df['path'].apply(lambda x: x.split('/')[0])

print(train_df.head())  # Check that it has 'path' and 'label' columns

print("Loading validation data into the Data Frame..")
val_df = pd.read_csv('../../dataset/raw/Food-101/meta/verified_val.tsv', sep='\t', header=None, names=['path', 'label'])
val_df = val_df[val_df['label'] == 1]  # Use only verified images
val_df['label'] = val_df['path'].apply(lambda x: x.split('/')[0])

print(val_df.head())  # Check that it has 'path' and 'label' columns

print("Making training Data Generator from Data Frame for Training data")
train_generator = data_generator.flow_from_dataframe(
    train_df,
    directory=main_path,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes
)

print("Making training Data Generator from Data Frame for Validation data")
val_generator = data_generator.flow_from_dataframe(
    val_df,
    directory=main_path,
    x_col='path',
    y_col='label',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=classes
)

print("Clearing previous session ..")
kb.clear_session()

print("Defining the model..")
pre_trained_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299,299,3)))
tensor = pre_trained_model.output
tensor = GlobalAveragePooling2D()(tensor)
tensor = Dense(4096)(tensor)
tensor = BatchNormalization()(tensor)
tensor = Activation('relu')(tensor)
tensor = Dropout(0.5)(tensor)

predictions = Dense(len(classes), activation='softmax')(tensor)

model = Model(inputs=pre_trained_model.input, outputs = predictions)
for layer in pre_trained_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch = len(train_df) // batch_size
validation_steps = len(val_df) // batch_size

print("First pass..")
checkpoint = ModelCheckpoint(filepath=saved_path_first, verbose=1, save_best_only=True)
csv_logger = CSVLogger('first.3.log')

history = model.fit(train_generator,
    validation_data=val_generator,
    validation_steps = validation_steps,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    verbose=1,
    callbacks=[csv_logger, checkpoint]
    )


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.savefig('first_pass.png')


for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

print("Second pass..")
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath=saved_path_second, verbose=1, save_best_only=True)
csv_logger = CSVLogger('second.3.log')

history = model.fit(train_generator,
    validation_data=val_generator,
    validation_steps = validation_steps,
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    verbose=1,
    callbacks=[csv_logger, checkpoint]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.savefig('second_pass.png')
