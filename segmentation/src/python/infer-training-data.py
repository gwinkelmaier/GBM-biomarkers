'''Create inference images for the training data as a validation step.'''
import tensorflow as tf
from pathlib import Path
from unet import define_model
import argparse
from skimage import io, segmentation
import numpy as np

# GLOBALS
HOME_DIR = Path.cwd().parent.parent
MODEL_DIR = HOME_DIR / 'models'
DATA_DIR = HOME_DIR / 'data/tfrecords'
SAVE_DIR = HOME_DIR / 'data/inference'
BATCH_SIZE = 16

# FUNCTIONS
def _get_dataset():
    '''Read TFRecords for Training.'''
    files = [str(i) for i in DATA_DIR.glob('*.tfrecord.gz')]
    ds = tf.data.TFRecordDataset(files, compression_type='GZIP')\
            .map(_get_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def _get_example(proto):
    '''Convert a TFRecord into training data.'''
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        }
    features = tf.io.parse_single_example(proto, feature_description)
    I = tf.io.decode_raw(features['image'], out_type=tf.float64)
    I = tf.reshape(I, [224, 224, 3])
    M = tf.io.decode_raw(features['mask'], out_type=tf.float64)
    M = tf.reshape(M, [224, 224, 2])
    return I, M

def custom_loss(alpha):
    def compute_loss(y_true, y_pred):
        m = tf.convert_to_tensor(y_true[:, :, :, 0])
        w = tf.convert_to_tensor(y_true[:, :, :, 1])
        y_pred = tf.convert_to_tensor(y_pred)
        y1 = tf.keras.losses.sparse_categorical_crossentropy(m, y_pred)
        y2 = tf.math.multiply(w, y1)
        y1 = tf.reduce_mean(y1)
        y2 = tf.reduce_mean(y2)
        return y1 + alpha*y2
    return compute_loss

def main():
    '''Main entrypoint.'''
    # Input Arguments
    parser = argparse.ArgumentParser('infer-training-data.py', description='Create overlaid inference images from training data.')
    parser.add_argument('MODEL', help='Name of model to use for inference')
    flags = vars(parser.parse_args())

    # Create Inference Data
    ds = _get_dataset()

    # Create Model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.models.load_model(str(MODEL_DIR/flags['MODEL']), custom_objects={'compute_loss': custom_loss(10)})

    # Infer Model
    print(f"Saving predictions to '{SAVE_DIR}'")
    count = 0
    for elem in ds.batch(BATCH_SIZE):
        y = model.predict(elem[0])
        y = np.where(y>0.5, 1, 0)
        for pred, image, mask in zip(y, elem[0].numpy(), elem[1].numpy()):
            savename = SAVE_DIR / f"prediction{count:04d}.png"
            save_image = segmentation.mark_boundaries(image, np.where(mask[:, :, 0]>0.5, 1, 0), color=[0, 1, 0])
            save_image = segmentation.mark_boundaries(save_image, pred[:, :, 1], color=[1, 1, 0])
            io.imsave(savename, save_image)
            count += 1


if __name__ == "__main__":
    main()
