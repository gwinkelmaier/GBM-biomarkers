"""Create inference images for the training data as a validation step."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import io, segmentation

from unet import define_model

# GLOBALS
HOME_DIR = Path.cwd().parent.parent
MODEL_DIR = HOME_DIR / "models"
DATA_DIR = Path("/home/gwinkelmaier/MILKData/NCI-GDC/LGG/Tissue_slide/")
BATCH_SIZE = 64

# FUNCTIONS
def _convert_to_monochromatic(image, mask):
    """Convert RGB image into monochromaitc image."""

    def _map(image):
        I = image.numpy()
        return 100 * I[:, :, 2] / (I[:, :, 0] + I[:, :, 1])

    image = tf.py_function(_map, [image], tf.float64)
    image = tf.expand_dims(image, axis=-1)

    return (image, mask)


def _get_dataset(wsi_path, mono_flag):
    """Find Patched images and create a tensorflow dataset."""
    filenames = [
        str(i) for i in (wsi_path / "2021/STBpatches/").glob("**/*_tiles/*.png")
    ]
    if not mono_flag:
        ds = (
            tf.data.Dataset.from_tensor_slices(filenames)
            .map(_get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
    else:
        ds = (
            tf.data.Dataset.from_tensor_slices(filenames)
            .map(_get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(_convert_to_monochromatic)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    return ds


def _get_image(filename):
    """Convert filename to image as a part of TF pipeline."""
    raw = tf.io.read_file(filename)
    image = tf.io.decode_png(raw)[:, :, :3]
    image = tf.cast(image, tf.float32) / 255.0
    return image, filename


def _background_filter(image, filename):
    """Filter images that are >50% background and/or all black."""
    I_white = tf.math.reduce_sum(tf.where(image > 0.9, 1, 0), axis=2)
    I_white = tf.math.reduce_sum(tf.where(I_white == 3, 1, 0))
    white_perct = I_white / (224 * 224)

    I_black = tf.math.reduce_sum(tf.where(image == 0, 1, 0), axis=2)
    I_black = tf.math.reduce_sum(tf.where(I_black == 3, 1, 0))
    black_perct = I_black / (224 * 224)

    if (white_perct > 0.5) or (black_perct > 0.5):
        return False
    else:
        return True


def custom_loss(alpha):
    def compute_loss(y_true, y_pred):
        m = tf.convert_to_tensor(y_true[:, :, :, 0])
        w = tf.convert_to_tensor(y_true[:, :, :, 1])
        y_pred = tf.convert_to_tensor(y_pred)
        y1 = tf.keras.losses.sparse_categorical_crossentropy(m, y_pred)
        y2 = tf.math.multiply(w, y1)
        y1 = tf.reduce_mean(y1)
        y2 = tf.reduce_mean(y2)
        return y1 + alpha * y2

    return compute_loss


def main():
    """Main entrypoint."""
    # Input Arguments
    parser = argparse.ArgumentParser(
        "infer-wsi.py",
        description="Create overlaid inference images from training data.",
    )
    parser.add_argument("MODEL", help="Name of model to use for inference")
    parser.add_argument("WSI", help="Name of WSI to make predictions on")
    parser.add_argument(
        "-s",
        "--save",
        dest="SAVE_DIR",
        default=None,
        help="Name of save directory (default=MODEL)",
    )
    parser.add_argument(
        "-m",
        "--mono",
        dest="MONO",
        action="store_true",
        help="monochromatic pre-processing flag (model dependent)",
    )
    flags = vars(parser.parse_args())

    # Create SAVE_DIR if omitted
    print(flags["SAVE_DIR"])
    if flags["SAVE_DIR"] is None:
        flags["SAVE_DIR"] = flags["MODEL"]

    # Create Inference Data
    ds = _get_dataset(DATA_DIR / flags["WSI"], flags["MONO"])
    # for elem in ds.batch(BATCH_SIZE).take(1):
    #     f = Path(elem[1].numpy()[0].decode('utf-8'))
    #     print(elem[0].numpy().shape, end = '\t')
    #     print(elem[0].numpy().min(), end = '\t')
    #     print(elem[0].numpy().max())
    #     plt.imshow(elem[0].numpy()[0])
    #     plt.title(f.name)
    #     plt.show()

    # exit()

    # Create Model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.models.load_model(
            str(MODEL_DIR / flags["MODEL"]),
            custom_objects={"compute_loss": custom_loss(10)},
        )

    ## Infer Model
    # Create a Save Directory
    save_dir = DATA_DIR / flags["WSI"] / "2021" / flags["SAVE_DIR"]
    save_dir.mkdir(parents=True, exist_ok=True)
    # Cycle through dataset
    for elem in ds.batch(BATCH_SIZE):
        y = model.predict(elem[0])
        # Cycle Through Predictiosn
        for probability, filename in zip(y, elem[1].numpy()):
            # Save Image
            savename = Path(filename.decode("utf-8")).name
            save_image = probability[:, :, 1]
            try:
                io.imsave(str(save_dir / savename), save_image)
            except:
                save_image = np.clip(save_image, [0, 255])
                io.imsave(str(save_dir / savename), save_image)


if __name__ == "__main__":
    main()
