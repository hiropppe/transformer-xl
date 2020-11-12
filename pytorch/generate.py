# coding: utf-8
import argparse
import time
import math
import os
import sys
import itertools
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_utils import get_lm_corpus, Corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel


def load_model(model_dir, spm_file, cuda=False):
    global model, para_model, spm, device
    device = torch.device('cuda' if cuda else 'cpu')

    with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    para_model = model.to(device)

    import sentencepiece as sp
    spm = sp.SentencePieceProcessor()
    spm.Load(spm_file)

    n_all_param = sum([p.nelement() for p in model.parameters()])
    n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
    print('#params = {}'.format(n_all_param))
    print('#non emb params = {}'.format(n_nonemb_param))


def greedy(start_string, L=100):
    start = time.time()
    seq = spm.encode_as_ids(start_string)
    init_len = len(seq)
    dec_inp = torch.LongTensor(seq).view(-1, 1).to(device)
    while dec_inp.size(0) < init_len + L:
        tgt_len = dec_inp.size(0)
        mems = tuple()
        ret = para_model.logit(dec_inp, tgt_len, *mems)
        logit, mems = ret[0], ret[1:]
        if device.type == 'cuda':
            dec_out = torch.argmax(logit, dim=-1).view(tgt_len, -1).cpu().numpy()[-1:, 0].tolist()
        else:
            dec_out = torch.argmax(logit, dim=-1).view(tgt_len, -1).numpy()[-1:, 0].tolist()
        seq.extend(dec_out)
        dec_inp = torch.LongTensor(seq).view(-1, 1).to(device)
    elapsed = time.time() - start

    text = spm.decode(seq)
    print('Greedy:')
    print(f'  Text: {text}')
    print(f'  IDs: {seq}')
    print('  Elapsed {:.4f} sec ({:.4f} ms/token)'.format(elapsed, elapsed*1000/L))


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--start-string', type=str,
                        help='')
    parser.add_argument('--seq_len', '-L', type=int, default=100,
                        help='number of tokens to predict')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model_dir', default='LM-TFM', type=str,
                        help='model directory.')
    parser.add_argument('--spm_file', type=str,
                        help='path to spm file')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed_all(args.seed)

    print('=' * 100)
    for k, v in args.__dict__.items():
        print('    - {} : {}'.format(k, v))
    print('=' * 100)

    if not torch.cuda.is_available():
        args.cuda = False

    load_model(args.model_dir, args.spm_file, args.cuda)

    greedy(args.start_string, args.seq_len)


if __name__ == "__main__":
    main()
