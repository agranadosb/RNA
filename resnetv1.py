import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import GaussianNoise as GN
from tensorflow.keras.layers import Input, MaxPooling2D, AvgPool2D
from data import steps_per_epoch, train, test, num_classes, epochs

input_shape = (32, 32, 3)
gn_prob = 0.4
filters = [64, 128, 256]


def activation(x):
    y = BN()(x)
    y = GN(gn_prob)(y)
    y = Activation('relu')(y)
    return y


def block_identity(x, filter):
    y = Conv2D(filter, 3, 1, padding='same')(x)
    y = activation(y)
    y = Conv2D(filter, 3, 1, padding='same')(y)

    x = Add()([y, x])
    x = activation(x)

    return x

def block(x, filter):
    y = Conv2D(filter, 3, 2, padding='same')(x)
    y = activation(y)
    y = Conv2D(filter, 3, 1, padding='same')(y)

    x = Conv2D(filter, 1, 2, padding='same')(x)

    x = Add()([y, x])
    x = activation(x)

    return x


def dense_layer(x, n, activation):
    x = Dense(n)(x)
    x = BN()(x)
    x = GN(gn_prob)(x)
    x = Activation(activation)(x)
    return x


def conv_subnet(x):
    x = Conv2D(filters[0], 7, 2, padding='same')(x)
    x = MaxPooling2D(2)(x)

    x = block_identity(x, filters[0])
    x = block_identity(x, filters[0])

    for i in filters[1:]:
        x = block(x, i)
        x = block_identity(x, i)

    x = AvgPool2D(2)(x)

    x = Flatten()(x)

    return x


def input_subnet():
    x_input = Input(input_shape)
    x = BN()(x_input)
    return x, x_input


def dense_subnet(x):
    x = dense_layer(x, num_classes, 'softmax')
    return x


def build_network():
    x, x_input = input_subnet()
    x = conv_subnet(x)
    x = dense_subnet(x)

    return tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet")


model = build_network()

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
set_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.8, patience=10, min_lr=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])


@tf.function
def data_augmentation(image, tag):
    image = tf.image.random_flip_left_right(image)
    return image, tag


model.summary()
tf.keras.utils.plot_model(model)

model.fit(
    train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test,
    callbacks=[set_lr],
)

# TEST
scores = model.evaluate(test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
