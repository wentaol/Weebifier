import math

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch import nn
from torch.nn import functional as F

import numpy as np
import re

from .models import *

import os
package_directory = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Weebifier:
    def __init__(self, 
                 csv_path = os.path.join(package_directory, 
                                         "db.csv"),
                 net_path = os.path.join(package_directory, 
                                         "weebifier.pt")):
        # Load db
        self.db = {}
        if(os.path.exists(csv_path)):
            with open(csv_path, "r", encoding="utf-8") as f:
                headers = f.readline().strip().split(",")
                for line in f:
                    toks = line.strip().split(",")
                    self.db[toks[1]] = toks[2]
                print("dictionary size:", len(self.db))
        # Load configs
        config = torch.load(net_path)
        network_state = config["state_dict"]
        self.en_itos = config["en_itos"]
        self.jp_itos = config["jp_itos"]
        self.init_token = config["init_token"]
        self.eos_token = config["eos_token"]
        embed_size = config["embed_size"]
        hidden_size = config["hidden_size"]
        self.en_stoi = dict(zip(self.en_itos, range(len(self.en_itos))))
        self.jp_stoi = dict(zip(self.jp_itos, range(len(self.jp_itos))))
        en_size = len(self.en_itos)
        jp_size = len(self.jp_itos)
        self.jp_start_idx = self.jp_stoi[self.init_token]
        # Create network
        encoder = Encoder(en_size, embed_size, hidden_size,
                          n_layers=2, dropout=0.5)
        decoder = Decoder(embed_size, hidden_size, jp_size,
                          n_layers=2, dropout=0.5)
        seq2seq = Seq2Seq(encoder, decoder).to(device)
        seq2seq.load_state_dict(network_state)
        seq2seq.eval()
        self.net = seq2seq.to(device)

    def word2tensor(self, word):
        idxs = [self.en_stoi[c] 
                for c in self.init_token + word + self.eos_token]
        return torch.tensor(idxs).reshape(-1,1)

    def tensor2str(self, tensor):
        tmp = tensor[:,0].detach().cpu().numpy().astype(np.long)
        return "".join(self.jp_itos[idx] for idx in tmp)

    # assumes word has already been stripped and cleaned
    def weebify_word(self, word, max_input_len = 36, max_output_len=24,
                     use_db = True, use_net = True):
        if len(word) > max_input_len:
            return "ほにゃらら"
        elif use_db and word in self.db:
            return self.db[word]
        elif use_net and re.match("^[a-zA-Z]+$", word) != None:
            with torch.no_grad():
                x = self.word2tensor(word).to(device)
                tmpy = torch.zeros([max_output_len, 1], 
                                dtype=torch.long).to(device)
                # Idx for padding
                tmpy[0] = self.jp_start_idx
                output = self.net(x, tmpy, -1, inference=True)
                tmp = self.tensor2str(output)
                return tmp[1:tmp.index(self.eos_token)]
        else:
            return word

    def weebify(self, sentence, use_db=True, use_net=True):
        toks = re.findall(r"\w+|[^\w\s]", sentence.upper())
        jtoks = [self.weebify_word(tok.strip(), use_db=use_db, use_net=use_net) 
                 for tok in toks]
        return " ".join(jtoks)