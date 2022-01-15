import tensorflow as tf
from keras.utils import np_utils
from functools import partial
from albumentations import (
    Compose,
    RandomBrightness,
    HorizontalFlip,
    CoarseDropout,
    GaussNoise,
    CropAndPad,
    ShiftScaleRotate
)

AUTO = tf.data.AUTOTUNE
batch_size = 400
epochs = 100
num_classes = 10
transforms = Compose(
    [
        RandomBrightness(limit=0.1, p=0.5),
        CoarseDropout(max_holes=1, max_height=2, max_width=2, p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
    ]
)


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
