#!/usr/bin/env python
import numpy as np
import os
import sys

from pathlib import Path
from tqdm import tqdm


def main(path, valid_size=0.1, test_size=0.1):
    path = Path(path)
    f = open(path)
    docs = f.readlines()
    doc_size = len(docs)
    #idx = np.random.permutation(len(docs))
    idx = np.arange(doc_size)

    train_offset = int(doc_size*(1.0-valid_size-test_size))
    if test_size:
        valid_offset = train_offset + int(doc_size*valid_size)
    else:
        valid_offset = None

    with open(path.parent / 'train.txt', 'w') as train_out:
        for i in tqdm(idx[:train_offset]):
            doc = docs[i].strip()
            if doc:
                train_out.write(doc + '\n')

    with open(path.parent / 'valid.txt', 'w') as valid_out:
        if valid_offset:
            pbar = tqdm(idx[train_offset:valid_offset])
        else:
            pbar = tqdm(idx[train_offset:])

        for i in pbar:
            doc = docs[i].strip()
            if doc:
                valid_out.write(doc + '\n')

    if valid_offset:
        with open(path.parent / 'test.txt', 'w') as test_out:
            for i in tqdm(idx[valid_offset:]):
                doc = docs[i].strip()
                if doc:
                    test_out.write(doc + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
