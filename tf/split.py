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

    valid_offset = 0
    test_offset = 0

    if valid_size:
        valid_offset = int(doc_size*valid_size)
        with open(path.parent / 'valid.txt', 'w') as valid_out:
            for i in tqdm(idx[:valid_offset]):
                doc = docs[i].strip()
                #if not any(doc.endswith(p) for p in ['。', '！', '？']):
                #    doc += doc + '。'
                if doc:
                    valid_out.write(doc + '\n')

    if test_size:
        test_offset = int(doc_size*test_size)
        with open(path.parent / 'test.txt', 'w') as test_out:
            for i in tqdm(idx[valid_offset:valid_offset + test_offset]):
                doc = docs[i].strip()
                #if not any(doc.endswith(p) for p in ['。', '！', '？']):
                #    doc += doc + '。'
                if doc:
                    test_out.write(doc + '\n')

    with open(path.parent / 'train.txt', 'w') as train_out:
        for i in tqdm(idx[max(valid_offset, test_offset):]):
            doc = docs[i].strip()
            #if not any(doc.endswith(p) for p in ['。', '！', '？']):
            #    doc += doc + '。'
            if doc:
                train_out.write(doc + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
