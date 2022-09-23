'''Train a UNet architecture with potential a field loss function.'''
import tensorflow as tf
from pathlib import Path
from unet2 import define_model
import time
import argparse

# GLOBALS
HOME_DIR = Path.cwd().parent.parent
MODEL_DIR = HOME_DIR / 'models'
DATA_DIR = HOME_DIR / 'data/tfrecords'
SAVE_DIR = HOME_DIR / 'data/inference' 
BATCH_SIZE = 32

# FUNCTIONS
def _convert_to_monochromatic(image, mask):
    '''Convert RGB image into monochromaitc image.'''

    def _map(image):
        I = image.numpy()
        return 100 * I[:, :, 2] / (I[:, :, 0] + I[:, :, 1])
    image = tf.py_function(_map, [image], tf.float64)
    image = tf.expand_dims(image, axis=-1)

    return (image, mask)

def _get_plain_dataset(mono_flag:bool):
    '''Read TFRecords for Training.'''
    files = [str(i) for i in DATA_DIR.glob('*.tfrecord.gz')]
    if mono_flag:
        ds = tf.data.TFRecordDataset(files, compression_type='GZIP')\
                .map(_get_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .map(_convert_to_monochromatic)\
                .prefetch(tf.data.experimental.AUTOTUNE)
    else:
        ds = tf.data.TFRecordDataset(files, compression_type='GZIP')\
                .map(_get_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def _get_no_supp_dataset(mono_flag:bool):
    '''Read TFRecords for Training.'''
    files = [str(i) for i in DATA_DIR.glob('*.tfrecord.gz')]
    files = [i for i in files if 'cropped' not in i]
    if mono_flag:
        ds = tf.data.TFRecordDataset(files, compression_type='GZIP')\
                .map(_get_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .map(_convert_to_monochromatic)\
                .prefetch(tf.data.experimental.AUTOTUNE)
    else:
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
    # Get input arguments
    parser = argparse.ArgumentParser(prog='train-all.py',
                                     description='train on all available training data')
    parser.add_argument('-s', '--supp', dest='SUPP', action='store_true', 
                        help='flag to use supplementary images or not')
    parser.add_argument('-m', '--mono', dest='MONO', action='store_true', 
                        help='monochromatic pre-processing flag')
    flags = vars(parser.parse_args())

    # Create Training Data
    if flags['SUPP']:
        ds = _get_no_supp_dataset(flags['MONO'])
    else:
        ds = _get_plain_dataset(flags['MONO'])

    # Create Model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if not flags['MONO']:
            model = define_model([224, 224, 3])
        else:
            model = define_model([224, 224, 1])
        adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = custom_loss(10)
        model.compile(optimizer=adam, loss=loss)
    print(model.summary())

    # Train Model
    model.fit(ds.batch(BATCH_SIZE),
              epochs=50,
              verbose=1)

    # Save Model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.save(str(MODEL_DIR/timestamp))

if __name__ == "__main__":
    main()
