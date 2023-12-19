import torch
import torch.nn as nn
import torch.nn.functional as F

from model_gru import ConvGRU, ConvGRU_self, ConvGRU_other
from model_attention import TransformerEncoder

from torch.nn import GRU

class EmotionIC(nn.Module):
    def __init__(self, hidden_dim, output_dim,
                trans_n_layers, indi_n_layer,
                dropout=0.6, attn_drop=0.6, feed_drop=0.6, rnn_drop=0.6,
                use_dropout=False):
        super(EmotionIC, self).__init__()

        self.trans_n_layers = trans_n_layers
        self.indi_n_layer   = indi_n_layer

        dim = 1024

        self.embedding_dim = dim
        self.num_head = 8
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.dropout = dropout
        self.dropout_attn = attn_drop
        self.dropout_feed = feed_drop
        self.dropout_rnn =  rnn_drop

        after_norm = 1
        attn_type =  'convpos' #'convpos'       #'convpos'   'adatrans'     'transformer'    'convtrans'   'convother'     'convpos_seg'  'conv_order'
        pos_embed = None   # 'sin'  None  'fix'
        self.global_encoder = TransformerEncoder(self.trans_n_layers, self.embedding_dim, self.num_head,
                                                feedforward_dim= self.hidden_dim, dropout= self.dropout_feed,
                                                after_norm=after_norm, attn_type=attn_type,
                                                scale=attn_type=='adatrans', dropout_attn=self.dropout_attn,
                                                pos_embed=pos_embed, batch_first=False)
        
        self.conv_GRU_indi = ConvGRU(self.embedding_dim, self.hidden_dim, num_layers=self.indi_n_layer,
                                bidirectional = False,
                                dropout=self.dropout_rnn if use_dropout else 0.)                                

        self.LN_glob   = nn.LayerNorm( self.embedding_dim, elementwise_affine=True )
        self.LN_local  = nn.LayerNorm(    self.hidden_dim, elementwise_affine=True )
        self.LN_origin = nn.LayerNorm(    self.embedding_dim, elementwise_affine=True )

        self.fc_embed = nn.Linear(1*self.hidden_dim + 2*self.embedding_dim, self.hidden_dim)

        self.fc_out = nn.Sequential(
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_dim, self.output_dim)
                        )
        

    def forward(self, text, umask, qmask):

        glob_hidden = self.global_encoder(text, umask, qmask)
        indi_hidden = self.conv_GRU_indi(text, qmask, umask)
        text        = self.LN_origin(text)
        fc_embeds = self.fc_embed(torch.cat((   self.LN_glob(glob_hidden), 
                                                self.LN_local(indi_hidden), 
                                                text), dim=-1))
        fc_out = self.fc_out(fc_embeds)       

        return F.log_softmax(fc_out, 2), fc_embeds