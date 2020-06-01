from torchtext.data import Field, BucketIterator, TabularDataset
import os
import math
import random

import numpy as np

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.nn import functional as F

from weebifier.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# this line is only needed if cudnn crashes
torch.backends.cudnn.enabled = False

def tokenize(text):
    return [c for c in text]

def load_dataset(path="./data", train_csv="train.csv", val_csv="val.csv", 
                 init_token='^', eos_token='$', batch_size=32):
    INDEX = Field(sequential=False, 
                  use_vocab=False, 
                  pad_token=None, 
                  unk_token=None)
    EN = Field(tokenize=tokenize, 
               include_lengths=True,
               init_token=init_token, 
               eos_token=eos_token)
    JP = Field(tokenize=tokenize, 
               include_lengths=True,
               init_token=init_token, 
               eos_token=eos_token, 
               is_target=True)
    FREQ = Field(sequential=False, 
                   use_vocab=False, 
                   pad_token=None, 
                   unk_token=None,
                   dtype=torch.float32)
    data_fields = [('index', INDEX), ('english', EN), 
                   ('japanese', JP), ('frequency', FREQ)]
    train, val = TabularDataset.splits(path=path, 
                                       train=train_csv, 
                                       validation=val_csv, 
                                       skip_header = True,
                                       format='csv', fields=data_fields)
    EN.build_vocab(train.english)
    JP.build_vocab(train.japanese)
    train_iter, val_iter = BucketIterator.splits((train, val),
                                                 batch_size=batch_size, 
                                                 sort=False,
                                                 repeat=False)
    return train_iter, val_iter, EN, JP

# Evaluate
def evaluate(model, val_iter, vocab_size, EN, JP):
    model.eval()
    pad = JP.vocab.stoi['<pad>']
    total_loss = 0
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            src, len_src = batch.english
            trg, len_trg = batch.japanese
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                                   trg[1:].contiguous().view(-1),
                                   ignore_index=pad)
            total_loss += loss.item()
        return total_loss / len(val_iter)

# Train 1 epoch
def train(e, model, optimizer, train_iter, vocab_size, grad_clip, EN, JP, 
          weighted_loss = False):
    model.train()
    total_loss = 0
    pad = JP.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        #print(batch)
        src, len_src = batch.english
        trg, len_trg = batch.japanese
        src, trg = src.to(device), trg.to(device)
        freq = torch.clamp(1e4 * batch.frequency.to(device), 0.1, 10.0)
        optimizer.zero_grad()
        output = model(src, trg)
        if weighted_loss:
            loss = F.nll_loss(output[1:].permute(1,2,0),
                              trg[1:].permute(1,0),
                              ignore_index=pad,
                              reduction='none')
            loss = torch.sum(loss, dim=1)
            loss = torch.mean(freq * loss)
        else:
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                           trg[1:].contiguous().view(-1),
                           ignore_index=pad)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        if b % 500 == 0 and b != 0:
            total_loss = total_loss / 500
            print("[%d][loss:%5.3f]" %
                  (b, total_loss))
            total_loss = 0

# Save model
def save_model(EN, JP, model, embed_size, hidden_size, path):
    conf = {"en_itos": EN.vocab.itos, 
            "jp_itos": JP.vocab.itos,
            "init_token": JP.init_token,
            "eos_token": JP.eos_token,
            "embed_size" : embed_size,
            "hidden_size" : hidden_size,
            "state_dict" : seq2seq.state_dict()}
    torch.save(conf, path)

if __name__ == "__main__":
    batch_size = 64
    hidden_size = 256
    embed_size = 128
    lr = 3e-4
    epochs = 20
    grad_clip = 10.0
    use_weighted_loss = False
    datadir = "data"
    outdir = "checkpoints/unweighted_2"

    print("[!] preparing dataset...")
    train_iter, val_iter, EN, JP = load_dataset(path=datadir,
                                                batch_size=batch_size)
    en_size, jp_size = len(EN.vocab), len(JP.vocab)
    print("[TRAIN]:%d (dataset:%d)"
        % (len(train_iter), len(train_iter.dataset)))
    print("[VAL]:%d (dataset:%d)"
        % (len(val_iter), len(val_iter.dataset)))
    print("[EN_vocab]:%d [JP_vocab]:%d" % (en_size, jp_size))

    print("[!] Instantiating models...")
    encoder = Encoder(en_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, jp_size,
                      n_layers=2, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=lr)

    print("[!] Training...")
    best_val_loss = None
    for e in range(epochs):
        train(e, seq2seq, optimizer, train_iter, jp_size, 
              grad_clip, EN, JP, 
              weighted_loss=use_weighted_loss)
        val_loss = evaluate(seq2seq, val_iter, jp_size, EN, JP)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.3f"
            % (e, val_loss, math.exp(val_loss)))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            save_model(EN, JP, seq2seq, embed_size, hidden_size, 
                       outdir + '/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
