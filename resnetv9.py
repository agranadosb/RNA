import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add, Dropout
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import GaussianNoise as GN
from tensorflow.keras.layers import Input, MaxPooling2D, AvgPool2D
from data import steps_per_epoch, train, test, num_classes, epochs

input_shape = (32, 32, 3)
gn_prob = 0.3
dp_prob = 0.3
filters = [32, 64, 128, 256]


def activation(x):
    y = BN()(x)
    y = GN(gn_prob)(y)
    y = Activation("relu")(y)
    return y


def dense_layer(x, n, activation):
    x = Dense(n)(x)
    x = BN()(x)
    x = GN(gn_prob)(x)
    x = Activation(activation)(x)
    return x


def layer(x, filters):
    x = Conv2D(filters, 3, strides=1, padding="same")(x)
    x = activation(x)

    return x


def layer_residual(x, filters_output):
    input_tensor = layer(x, filters_output)
    input_tensor = MaxPooling2D(2)(input_tensor)

    x = layer(input_tensor, filters_output)
    x = layer(x, filters_output)

    out = Add()([x, input_tensor])

    return out


def conv_subnet(x):
    x = layer(x, 64)
    x = layer_residual(x, 128)

    x = layer(x, 256)
    x = MaxPooling2D(2)(x)
    x = layer_residual(x, 512)

    x = MaxPooling2D(4)(x)

    x = Flatten()(x)

    return x


def input_subnet():
    x_input = Input(input_shape)
    x = BN()(x_input)
    return x, x_input


def dense_subnet(x):
    x = dense_layer(x, num_classes, "softmax")
    return x


def build_network():
    x, x_input = input_subnet()
    x = conv_subnet(x)
    x = dense_subnet(x)

    return tf.keras.models.Model(inputs=x_input, outputs=x, name="ResNet")


model = build_network()

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
set_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=10, min_lr=1e-6)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


model.summary()
tf.keras.utils.plot_model(model)

model.fit(
    train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=test,
    callbacks=[set_lr],
)

scores = model.evaluate(test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
