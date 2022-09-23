'''Run Tiles through pen-mark model and save predcitions.'''
import argparse
import tensorflow as tf
from pathlib import Path
import pandas as pd

# GLOBALS
DATA_DIR = Path('/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image')
MODEL_DIR = Path.cwd().parent / 'models'

# FUNCTIONS
def tf_dataset(list_files):
    '''Create a TF Dataset from a list of file names'''
    ds = tf.data.Dataset.from_tensor_slices(list_files).map(_input_mapper,
             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds

def _input_mapper(filename):
    '''Map a Filename into an image.'''
    raw = tf.io.read_file(filename)
    I = tf.io.decode_image(raw)
    I = tf.cast(I, tf.float64) / 255.0
    I = 2.0 * I - 1.0
    return I

def main():
    # Get Input Args
    parser = argparse.ArgumentParser(prog='infer-pen.py', description='Inference of pen-mark model')
    parser.add_argument('--wsi', dest='WSI', required=True, help='Name of WSI folder')
    parser.add_argument('--model', dest='MODEL', default='pen', help='Name of Pen Model folder')
    flags = vars(parser.parse_args())

    # Get a list of tile names
    image_names = [str(i) for i in (DATA_DIR/flags['WSI']/'2021/patches/').glob('**/*_tiles/*.png')]
    print(len(image_names))
    print(image_names[0])

    # Get MetaData File
    meta_name = (DATA_DIR/flags['WSI']/'2021/patches').glob('**/tile_selection.tsv')
    meta_df = pd.read_csv(next(meta_name), sep='\t')
    meta_df = meta_df[meta_df['Keep']==1]
    print(meta_df.head())

    # Read Images as DS
    ds = tf_dataset(image_names)
    for elem in ds.take(1):
        print(elem.shape)
        print(elem.numpy().min(), elem.numpy().max())

    # Load model
    model = tf.keras.models.load_model(str(MODEL_DIR/flags['MODEL']))

    # Make inference
    logits = model.predict(ds.batch(128).prefetch(tf.data.experimental.AUTOTUNE))
    print(logits.shape)
    print(logits[:10])

    # Save predictions
    meta_df['tissue_prob'] = logits[:, 0]
    meta_df['pen_prob'] = logits[:, 1]
    meta_df.to_csv(str(DATA_DIR/flags['WSI']/'2021/pen-marks.csv'), index=False)

# MAIN
if __name__ == "__main__":
    main()
