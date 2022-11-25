"""
Refer to detr.
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones
from typing import Optional


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leakyrelu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="gelu"):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=d_model, out_channels=2 * d_model, kernel_size=5),
                                   nn.BatchNorm1d(2 * d_model), nn.MaxPool1d(2), nn.GELU())  # ((300-5)/1+1)/2=148
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=2 * d_model, out_channels=d_model, kernel_size=5, stride=2),
            nn.BatchNorm1d(d_model), nn.GELU())  # (148-1)*2+5=299
        self.tconv11 = nn.Sequential(nn.Conv1d(in_channels=2 * d_model, out_channels=d_model, kernel_size=5),
                                     nn.BatchNorm1d(d_model), nn.GELU())  # (299-5)/1+1=295

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # q = k = v = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, v, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # src2 = self.norm1(src2)
        # src = src + self.dropout(src2)
        src = src.permute(1, 2, 0)
        src2 = self.conv1(src)
        src2 = self.tconv1(src2)
        src2 = torch.concat((src[:, :, :-1], src2), dim=1)
        src2 = self.tconv11(src2)
        src = self.dropout1(src2.permute(2, 0, 1))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.norm2(src2)
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="gelu"):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(int(d_model / 2), d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, int(d_model / 2))
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=int(d_model / 2), kernel_size=5, padding=2),
            nn.BatchNorm1d(int(d_model / 2)), nn.GELU())

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(int(d_model / 2))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # q = k = v = self.with_pos_embed(tgt, pos)
        tgt2 = self.multihead_attn(query=tgt,
                                   key=memory,
                                   value=memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.norm2(tgt2)
        tgt = tgt + self.dropout1(tgt2)
        # q = k = v = tgt
        # tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt2 = self.norm1(tgt2)
        # tgt = tgt + self.dropout(tgt2)
        tgt2 = self.conv1(tgt.permute(1, 2, 0))
        tgt = self.dropout2(tgt2.permute(2, 0, 1))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt2 = self.norm3(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        # output_list = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            # output_list.append(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        return output


class SMILESformer(nn.Module):

    def __init__(self, d_model=128, nhead=1, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=256, dropout=0.1, seq_len=400,
                 activation="gelu", normalize_before=False):
        super().__init__()
        self.seq_len = seq_len
        self.rnn = nn.LSTM(d_model, int(d_model / 2), 1, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.bos = nn.Parameter(torch.Tensor(seq_len, 1, d_model), requires_grad=True)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        # encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        # decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.layer = nn.Sequential(nn.Linear(int(seq_len * d_model / 2), int(seq_len * d_model / 16)),
                                   nn.LayerNorm(int(seq_len * d_model / 16)),
                                   nn.LeakyReLU(),
                                   nn.Dropout(dropout),
                                   nn.Linear(int(seq_len * d_model / 16), seq_len),
                                   nn.LayerNorm(seq_len))
        # self.trainable_embed = PositionEmbeddingLearned(d_model)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.4)
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # LxNxC
        # sin_pos = self.sin_pos_embed(src.size(0), self.d_model, src.device)
        # trainable_pos = self.trainable_embed(src)
        # pos = sin_pos + trainable_pos
        src2, _ = self.rnn(src)
        src2 = self.norm(src2)
        src = src + self.dropout(src2)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos=None)
        batch_size = memory.size(1)
        hs = self.decoder(self.bos.repeat(1, batch_size, 1), memory, tgt_mask=tgt_mask,
                          memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask,
                          pos=None, query_pos=None)
        hs = hs.permute(1, 0, 2).flatten(1)
        hs = self.layer(hs)
        return hs

    def sin_pos_embed(self, seq_len, d_model, device, max_len=1500):
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # max_len, 1, d_model
        return pe[:seq_len, :]


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, d_model=32):
        super().__init__()
        self.pos_embed = nn.Embedding(20, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        seq_len = x.shape[0]
        i = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embed(i)  # seq_len, dim
        pos = pos_emb.unsqueeze(1)  # seq_len, 1, dim
        return pos


class DDIModel(nn.Module):

    def __init__(self, args):
        super(DDIModel, self).__init__()
        self.pcba_dim = 128
        self.hid_dim = args.hid_dim
        self.vec_dim = args.vec_dim
        self.kge_dim = args.kge_dim
        self.smiles_max_len = args.smiles_max_len
        self.rel_total = args.rel_total
        self.conv0 = nn.Sequential(
            nn.Conv1d(in_channels=self.hid_dim, out_channels=self.hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hid_dim), nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.vec_dim, out_channels=self.hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hid_dim), nn.MaxPool1d(4), nn.LeakyReLU(), nn.Dropout(args.dropout))
        self.linear_kge = nn.Sequential(nn.Linear(self.kge_dim, int(self.kge_dim / 2)),
                                        nn.BatchNorm1d(int(self.kge_dim / 2)),
                                        nn.LeakyReLU(),
                                        nn.Dropout(args.dropout),
                                        nn.Linear(int(self.kge_dim / 2), self.hid_dim))
        self.linear_weave = nn.Sequential(nn.Linear(self.pcba_dim, int(self.pcba_dim / 2)),
                                          nn.BatchNorm1d(int(self.pcba_dim / 2)),
                                          nn.LeakyReLU(),
                                          nn.Dropout(args.dropout),
                                          nn.Linear(int(self.pcba_dim / 2), self.hid_dim))
        self.linear_mpnn = nn.Sequential(nn.Linear(self.pcba_dim, int(self.pcba_dim / 2)),
                                         nn.BatchNorm1d(int(self.pcba_dim / 2)),
                                         nn.LeakyReLU(),
                                         nn.Dropout(args.dropout),
                                         nn.Linear(int(self.pcba_dim / 2), self.hid_dim))
        self.linear_afp = nn.Sequential(nn.Linear(self.pcba_dim, int(self.pcba_dim / 2)),
                                        nn.BatchNorm1d(int(self.pcba_dim / 2)),
                                        nn.LeakyReLU(),
                                        nn.Dropout(args.dropout),
                                        nn.Linear(int(self.pcba_dim / 2), self.hid_dim))
        self.self_attn1 = nn.MultiheadAttention(self.hid_dim, 1, dropout=args.dropout)
        self.norm1 = nn.LayerNorm(self.hid_dim)
        self.dropout1 = nn.Dropout(args.dropout)
        self.linear11 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.dropout11 = nn.Dropout(args.dropout)
        self.linear12 = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.norm12 = nn.LayerNorm(self.hid_dim)
        self.dropout12 = nn.Dropout(args.dropout)
        self.activation1 = _get_activation_fn("gelu")

        self.self_attn2 = nn.MultiheadAttention(self.hid_dim, 1, dropout=args.dropout)
        self.norm2 = nn.LayerNorm(self.hid_dim)
        self.dropout2 = nn.Dropout(args.dropout)
        self.layer2 = nn.Sequential(nn.Linear(58 * self.hid_dim, 10 * self.hid_dim),
                                    nn.LayerNorm(10 * self.hid_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(args.dropout),
                                    nn.Linear(10 * self.hid_dim, self.rel_total))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.1)

    def forward(self, weaves, mpnns, afps, kges, vecs):  # vecs(batch*2*100*64)
        kges_l = self.linear_kge(kges[:, 0])
        kges_r = self.linear_kge(kges[:, 1])
        kges = torch.stack((kges_l, kges_r), 0)
        vecs = self.conv(torch.concat((vecs[:, 0], vecs[:, 1]), dim=1).permute(0, 2, 1)).permute(2, 0, 1)  # len=100*2/2=100
        q = k = v = vecs
        hids = self.self_attn1(q, k, v)[0]
        hids = self.norm1(hids)
        vecs = self.dropout1(hids)
        hids = self.linear12(self.dropout11(self.activation1(self.linear11(vecs))))
        hids = self.norm12(hids)
        hids = vecs + self.dropout12(hids)
        weaves_l = self.linear_weave(weaves[:, 0])
        weaves_r = self.linear_weave(weaves[:, 1])
        mpnns_l = self.linear_mpnn(mpnns[:, 0])
        mpnns_r = self.linear_mpnn(mpnns[:, 1])
        afps_l = self.linear_afp(afps[:, 0])
        afps_r = self.linear_afp(afps[:, 1])
        gnns = self.conv0(torch.stack((weaves_l, weaves_r, mpnns_l, mpnns_r, afps_l, afps_r), dim=1).permute(0, 2, 1)).permute(2, 0, 1)
        a = s = d = torch.concat((kges, gnns, hids), 0)
        gnns = self.self_attn2(a, s, d)[0]
        gnns = self.norm2(gnns)
        gnns = self.dropout2(gnns)

        gnns = gnns.permute(1, 0, 2).flatten(1)
        gnns = self.layer2(gnns)

        outputs = gnns
        return outputs
