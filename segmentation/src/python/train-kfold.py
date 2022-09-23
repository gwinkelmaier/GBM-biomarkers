'''Train a UNet architecture with potential a field loss function.'''
import tensorflow as tf
from pathlib import Path
from unet2 import define_model
import time
import numpy as np
from skimage import io, segmentation

# GLOBALS
HOME_DIR = Path.cwd().parent.parent
MODEL_DIR = HOME_DIR / 'models'
DATA_DIR = HOME_DIR / 'data/tfrecords'
SAVE_DIR = HOME_DIR / 'data/inference' 
BATCH_SIZE = 16
K = 5

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
    # Create Training Data
    ds = _get_dataset()

    # Create Model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = define_model([224, 224, 3])
        adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = custom_loss(10)
    model.compile(optimizer=adam, loss=loss)

    # Train Model
    for i in range(K):
        # Clean Training Data between iterations
        tr_ds = None

        # Assign Training/testing data
        for j in range(K):
            if i==j:
                te_ds = ds.shard(num_shards=K, index=j)
            elif tr_ds is None:
                tr_ds = ds.shard(num_shards=K, index=j)
            else:
                tr_ds = tr_ds.concatenate(ds.shard(num_shards=K, index=j))

        # Train model for this iteration
        model.fit(tr_ds.batch(BATCH_SIZE),
                  validation_data=te_ds.batch(BATCH_SIZE),
                  epochs=50,
                  verbose=2)

        # Save Predictions
        fold_dir = SAVE_DIR / f"Fold-{i}"
        if not fold_dir.exists():
            fold_dir.mkdir()
        count = 0
        for elem in te_ds.batch(BATCH_SIZE):
            y = model.predict(elem[0])
            y = np.where(y>0.5, 1, 0)
            for pred, image, mask in zip(y, elem[0].numpy(), elem[1].numpy()):
                savename = fold_dir / f"prediction{count:04d}.png"
                save_image = segmentation.mark_boundaries(image, np.where(mask[:, :, 0]>0.5, 1, 0), color=[0, 1, 0])
                save_image = segmentation.mark_boundaries(save_image, pred[:, :, 1], color=[1, 1, 0])
                io.imsave(savename, save_image)
                count += 1

if __name__ == "__main__":
    main()
