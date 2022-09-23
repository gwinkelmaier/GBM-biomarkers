'''Create TFRecords for training.

Uses Project data directory for locating files and saves to static 'tfrecords' folder.
'''
import sys
from pathlib import Path
import tensorflow as tf
from skimage.io import imread
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
import numpy as np

from weight_map import _compute_potential

#### GLOBALS
HOME_DIR = Path.cwd().parent.parent
DATA_DIR = HOME_DIR / 'data'
SAVE_DIR = DATA_DIR / 'tfrecords'

#### Functions
def _get_image(filename):
    '''Read Image and corresponding mask.

    Tests that the correct 'tif' file is read
        If Image/Mask pair is to be skipped, then return (None, None)
    '''
    # If Filetype is TIFF - only read cropped versions
    filetype = str(filename).split('.')[-1]
    if (filetype == 'tif') & ~('cropped' in str(filename)):
        print(f"Skipping Image '{filename}'")
        return (None, None)

    # Use PIL plugin to read tiff
    if filetype == 'tif':
        plugin='pil'
    else:
        plugin=None


    # Read Image
    I = imread(filename, plugin=plugin)

    # Read Mask
    maskname = str(filename).replace('images','masks')
    try:
        M = imread(maskname, plugin=plugin)
    except:
        maskname = maskname.replace('image','mask')
        M = imread(maskname, plugin=plugin)

    # Normalize Images
    if I.max() > 1.0:
        I = (I/255).astype(np.float32)
    if M.max() > 1.0:
        M = (M/255).astype(np.float32)
    return (I, M)

def _windowing(I, M, P):
    '''Tile the images into patches of 224x224.'''
    # Variables
    window_size = (224, 224, 9)
    step_size = (112, 112, 9)

    # Stack Image, mask, & potential field
    M = np.expand_dims(M, axis=2)
    M = np.tile(M , (1,1,3))
    P = np.expand_dims(P, axis=2)
    P = np.tile(P , (1,1,3))
    image_stack = np.concatenate([I, M, P], axis=2)

    # Sliding window based on 'variables'
    slices = view_as_windows(image_stack,
                             window_size,
                             step_size)

    # Return a single stack of all images
    output_stack = np.reshape(slices, [-1, window_size[0], window_size[1], window_size[2]])
    output_stack = np.concatenate([output_stack[:,:,:,:3],
                                   np.expand_dims(output_stack[:,:,:,3], axis=-1),
                                   np.expand_dims(output_stack[:,:,:,6], axis=-1)],
                                 axis=3)
    return output_stack

def _augmentation(image_stack):
    '''Augment Images.

    Flipping
    '''
    # output copy
    output_stack = image_stack

    # Cycle through windows
    for image in image_stack:
        # Horizontal Flip
        augment = np.expand_dims(image[::-1, :, :], axis=0)
        output_stack = np.concatenate([output_stack, augment], axis=0)

        # Vertical Flip
        augment = np.expand_dims(image[:, ::-1, :], axis=0)
        output_stack = np.concatenate([output_stack, augment], axis=0)

        # Combined Flip
        augment = np.expand_dims(image[::-1, ::-1, :], axis=0)
        output_stack = np.concatenate([output_stack, augment], axis=0)
    return output_stack

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _parse_single_example(I, M):
    """Serialize a single image and label"""
    data = {
        'image': _bytes_feature(I.tobytes()),
        'mask': _bytes_feature(M.tobytes()),
    }
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def create_tfrecord(image_stack, filename):
    '''Convert image stack into a single tfrecord.'''
    print(f"Saving to '{filename}'")
    writer = tf.io.TFRecordWriter(filename, tf.io.TFRecordOptions(compression_type='GZIP'))
    for i in image_stack:
        I = i[:, :, :3]
        M = i[:, :, 3:]
        proto = _parse_single_example(I, M)
        writer.write(proto.SerializeToString())

def _get_example(proto):
    '''Convert a TFRecord into training data.'''
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        }
    features = tf.io.parse_single_example(proto, feature_description)
    I = tf.io.decode_raw(features['image'], out_type=tf.float64)
    I = tf.reshape(I, [224, 224, 3])
    mask = tf.io.decode_raw(features['mask'], out_type=tf.float64)
    mask = tf.reshape(mask, [224, 224, 2])
    M = mask[:, :, 0]
    P = mask[:, :, 1]
    return I, M, P

def main():
    '''Main Entrypoint.'''

    # Create the save directory
    if not SAVE_DIR.is_dir():
        SAVE_DIR.mkdir()

    # Cycle through all images
    for f in DATA_DIR.glob('**/images/*'):
        # Check if file has already been computed
        savename = str(SAVE_DIR / str(f.name).split('.')[0]) + '.tfrecord.gz'
        if Path(savename).exists():
            print(f"'{savename}' exists.  Skipping")

        # Read image and mask
        I, M = _get_image(f)
        if I is None:
            continue

        # Compute potential field according to MINA KHOSHDELI
        P = _compute_potential(M)

        # Tile image into blocks
        stack = _windowing(I, M, P)

        # Augment all images
        stack = _augmentation(stack)

        # Write TFRecord
        create_tfrecord(stack, savename)


def validate_record():
    static_name = str(SAVE_DIR / 'image1.tfrecord.gz')
    ds = tf.data.TFRecordDataset(static_name, compression_type='GZIP')\
            .map(_get_example)

    print('Validating Data:')
    print('\tSingle Example')
    for elem in ds.take(1):
        print(f"\t{elem[0].numpy().shape} -> {elem[0].numpy().min()}, {elem[0].numpy().max()}")
        print(f"\t{elem[1].numpy().shape} -> {elem[1].numpy().min()}, {elem[1].numpy().max()}")
        print(f"\t{elem[2].numpy().shape} -> {elem[2].numpy().min()}, {elem[2].numpy().max()}")

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(elem[0].numpy())
    ax[1].imshow(elem[1].numpy(), cmap='gray')
    ax[2].imshow(elem[2].numpy(), cmap='gray')
    plt.show()

    print('\tBatched Example')
    for elem in ds.batch(10).take(1):
        print(f"\t{elem[0].numpy().shape} -> {elem[0].numpy().min()}, {elem[0].numpy().max()}")
        print(f"\t{elem[1].numpy().shape} -> {elem[1].numpy().min()}, {elem[1].numpy().max()}")
        print(f"\t{elem[2].numpy().shape} -> {elem[2].numpy().min()}, {elem[2].numpy().max()}")

    for N, elem in enumerate(ds):
        pass

    print(f"\tTotal Number of Images saved in 'image1.tfrecord.gz': {N+1}")


#### MAIN
if __name__ == "__main__":
    # main()
    validate_record()
