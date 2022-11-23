"""
Refer to detr.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
