from functools import partial

import tensorflow as tf
from albumentations import CoarseDropout, Compose, HorizontalFlip, RandomBrightness
from keras.utils import np_utils
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import GaussianNoise as GN
from tensorflow.keras.layers import Input, MaxPooling2D

AUTO = tf.data.AUTOTUNE
batch_size = 400
epochs = 100
num_classes = 10
input_shape = (32, 32, 3)
gn_prob = 0.0
dp_prob = 0.3
filters = [32, 64, 128, 256]
transforms = Compose(
    [
        RandomBrightness(limit=0.3, p=0.5),
        CoarseDropout(max_holes=2, max_height=2, max_width=2, p=0.5),
        HorizontalFlip(p=0.5),
    ]
)

"""
######################################
            Data functions
######################################
"""

@tf.function
def normalize(image, tag):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    return image, tag


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


@tf.function
def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two

    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)

    return (images, labels)


def aug_fn(image):
    data = {"image": image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.image.resize(aug_img, size=[32, 32])
    return aug_img


@tf.function
def apply_transformation(image):
    return tf.numpy_function(func=aug_fn, inp=[image], Tout=tf.float32)


@tf.function
def process_data(image, label):
    tensor = tf.map_fn(apply_transformation, image)
    return tensor, label

"""
#####################################
            Net functions
#####################################
"""

def activation(x):
    y = BN()(x)
    y = GN(gn_prob)(y)
    y = Activation("relu")(y)
    return y


def dense_layer(x, n, activation):
    x = Dense(n, kernel_regularizer=tf.keras.regularizers.L2(1e-3))(x)
    x = BN()(x)
    x = GN(gn_prob)(x)
    x = Activation(activation)(x)
    return x


def layer(x, filters):
    x = Conv2D(filters, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.L2(1e-3))(x)
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


"""
###############################
            Dataset
###############################
"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
steps_per_epoch = len(x_train) // batch_size


y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

train_ds_one = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .map(normalize)
    .shuffle(100)
    .repeat(epochs)
    .batch(batch_size)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .map(normalize)
    .shuffle(100)
    .repeat(epochs)
    .batch(batch_size)
)

train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))

test = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(normalize)
    .batch(batch_size)
)
train = train_ds.map(
    lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
).map(partial(process_data), num_parallel_calls=AUTO)


"""
#############################
            Model
#############################
"""
model = build_network()

opt = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=0.1)
set_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=7, min_lr=1e-4)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


model.summary()
tf.keras.utils.plot_model(model)

"""
################################
            Training
################################
"""
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
