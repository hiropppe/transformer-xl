import sys
import tensorflow_datasets as tfds

from tqdm import tqdm

lang = sys.argv[1]
split = sys.argv[2]
if len(sys.argv) > 3:
    outpath = sys.argv[3]
else:
    outpath = None

ds = tfds.load(f'wiki40b/{lang}', split=split)

def write(ds, out):
    for data in tqdm(ds.as_numpy_iterator()):
        text = data['text'].decode()
        print(text, file=out)

if outpath:
    with open(outpath, 'w') as out:
        write(ds, out)
else:
    write(ds, sys.stdout)


