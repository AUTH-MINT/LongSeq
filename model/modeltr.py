#import model.configtr as configtr
import torch
from torchcrf import CRF
import transformers
import torch.nn as nn
from model.config import Config
from model.vanilla_layers import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward
from model.linformer_layers import Linformer
from model.reformer_layers import Reformer
from model.longformer_layers import Longformer
import copy

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss

class EntityModel(nn.Module):
    def __init__(self, num_tag, tr=False):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.tr = tr

        self.N = Config.N
        self.h = Config.h
        self.k = Config.k
        self.d_model = Config.d_model
        self.d_ff = Config.d_ff
        self.trans_dropout = Config.trans_dropout
        self.max_len = Config.MAX_LEN

        ##Bert LM
        self.bert = Config.MODEL
        self.hidden_size = Config.hidden_size
        self.bert_drop_1 = nn.Dropout(Config.bert_droppout)

        ##Vanilla Layers
        self.c = copy.deepcopy
        self.attn = MultiHeadedAttention(self.h, self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.trans_dropout)
        self.vanilla = Encoder(EncoderLayer(self.d_model, copy.deepcopy(self.attn), copy.deepcopy(self.ff), self.trans_dropout), self.N)

        ##Linformer Layers
        #
        # self.linformer = Linformer(dim=self.d_model, seq_len=self.max_len, depth=self.N, heads=self.h, k=self.k )

        ##Reformer Layers

        # self.reformer = Reformer(dim=self.d_model, max_seq_len=self.max_len, depth=self.N, heads=self.h, lsh_dropout=0.1, causal=True)

        ##Longformer Layers
        self.longformer = Longformer(dim=self.d_model, depth=self.N, heads=self.h,  dropout=0, attention_window=self.N*[256], attention_dilation=self.N*[1], autoregressive=False, attention_mode='sliding_chunks')

        ##TransformerXL Layers

        # self.transformerXL =  MemTransformerLM(n_token=self.max_len, n_layer=self.N, n_head=self.h, d_model=self.d_model, d_inner=self.d_ff, dropout=self.trans_dropout, d_head=64,dropatt=0, mem_len=1600, tgt_len=0, ext_len=0)

        ##BiLSTM
        # self.lstm = nn.LSTM(self.d_model, bidirectional=True, batch_first=True)

        ##Boom_Layer
        # self.boom = Boom(d_model = self.d_model)

        self.out_tag = nn.Linear(self.d_model, self.num_tag)
        self.crf = CRF(self.num_tag, batch_first=True)


    def forward(self, ids, mask, token_type_ids, target_tag):
        o1 , _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.bert_drop_1(o1)

        # x, states = self.lstm(x)

        # for lstm
        # x = self.out_tag(x)

        #for linear
        # x = self.fc(x)

        #for transformers
        # x, _ = self.transformerXL(x)

        # x = self.vanilla(x, None)

        # x = self.linformer(x)

        # x = self.reformer(x)

        x = self.longformer(x)

        ##Boom Here
        # x = self.boom(x)

        # For crf
        emissions = self.out_tag(x)

        #Train and eval

        if self.tr is True:
            loss_tag = self.crf(emissions, target_tag)

            tag = self.crf.decode(emissions)
            tag = torch.LongTensor(tag)
            loss = -1*loss_tag
            return (loss, emissions, tag)

        # Testing
        else:
            tag = self.crf.decode(emissions)
            tag = torch.LongTensor(tag)
            return (tag, target_tag)
