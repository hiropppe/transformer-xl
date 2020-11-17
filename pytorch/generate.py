# coding: utf-8
import argparse
import time
import os
import sys

import numpy as np

import torch

from data_utils import Corpus


def load_model(model_dir, corpus_dir=None, spm_file=None, latest_model=False, cuda=False):
    global model, para_model, vocab, spm, device
    device = torch.device('cuda' if cuda else 'cpu')

    model_name = 'latest_model.pt' if latest_model else 'model.pt'

    with open(os.path.join(model_dir, model_name), 'rb') as f:
        model = torch.load(f)

    model.eval()
    para_model = model.to(device)

    if corpus_dir:
        corpus = torch.load(os.path.join(corpus_dir, 'cache.pt'))
        vocab = corpus.vocab
        spm = None
    else:
        import sentencepiece as sp
        spm = sp.SentencePieceProcessor()
        spm.Load(spm_file)
        vocab = None

    n_all_param = sum([p.nelement() for p in model.parameters()])
    n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])
    print('#params = {}'.format(n_all_param))
    print('#non emb params = {}'.format(n_nonemb_param))


def encode(context):
    if os.path.exists(context):
        if vocab:
            tensor = vocab.encode_file(context, ordered=True).view(-1, 1).to(device)
        else:
            encoded = []
            with open(context) as fin:
                for line in fin:
                    ids = spm.encode_as_ids(line)
                    encoded.append(torch.LongTensor(ids))
            tensor = torch.cat(encoded).view(-1, 1).to(device)
    else:
        if vocab:
            symbols = vocab.tokenize(context, add_eos=False)
            tensor = vocab.convert_to_tensor(symbols).view(-1, 1).to(device)
        else:
            ids = spm.encode_as_ids(context)
            tensor = torch.LongTensor(ids).view(-1, 1).to(device)
    return tensor


def decode(ids):
    if vocab:
        text = vocab.convert_to_sent(ids) 
    else:
        text = spm.decode(ids)
    return text


def greedy(context, L=100):
    start = time.time()
    dec_inp = encode(context)
    if device.type == 'cuda':
        seq = dec_inp.flatten().cpu().numpy().tolist()
    else:
        seq = dec_inp.flatten().numpy().tolist()
    init_len = dec_inp.size(0)
    while dec_inp.size(0) < init_len + L:
        tgt_len = dec_inp.size(0)
        mems = tuple()
        ret = model.logit(dec_inp, tgt_len, *mems)
        logit, mems = ret[0], ret[1:]
        if device.type == 'cuda':
            dec_out = torch.argmax(logit, dim=-1).view(tgt_len, -1).cpu().numpy()[-1:, 0].tolist()
        else:
            dec_out = torch.argmax(logit, dim=-1).view(tgt_len, -1).numpy()[-1:, 0].tolist()
        seq.extend(dec_out)
        dec_inp = torch.LongTensor(seq).view(-1, 1).to(device)
    elapsed = time.time() - start
    text = decode(seq[init_len:])
    print('Greedy:')
    print(f'  Text: {text}')
    #print(f'  IDs: {seq}')
    print('  Elapsed {:.4f} sec ({:.4f} ms/token)'.format(elapsed, elapsed*1000/L))


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
    parser.add_argument('--context', type=str,
                        help='')
    parser.add_argument('--seq_len', '-L', type=int, default=100,
                        help='number of tokens to predict')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--model_dir', default='LM-TFM', type=str,
                        help='model directory.')
    parser.add_argument('--corpus_dir', type=str,
                        help='')
    parser.add_argument('--spm_file', type=str,
                        help='path to spm file')
    parser.add_argument('--latest_model', action='store_true',
                        help='Load the latest model instead of the best model.')
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

    load_model(args.model_dir, args.corpus_dir, args.spm_file, args.latest_model, args.cuda)

    greedy(args.context, args.seq_len)


if __name__ == "__main__":
    main()
