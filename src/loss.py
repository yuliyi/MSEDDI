import torch
import torch.nn.functional as F


#
class KGELoss(torch.nn.Module):
    def __init__(self):
        super(KGELoss, self).__init__()

    def forward(self, output1, output2):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.sum(torch.pow(euclidean_distance, 2))
        return loss_contrastive


class SigmoidLoss(torch.nn.Module):

    def forward(self, p_scores, n_scores):
        p_loss = - F.logsigmoid(p_scores).mean()
        n_loss = - F.logsigmoid(-n_scores).mean()

        return (p_loss + n_loss) / 2, p_loss, n_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive