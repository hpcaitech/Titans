import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode


class embeddingLoss(nn.Module):

    def forward(self, train_iterator, args, model):

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach() *
                              F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = -positive_score.mean()
            negative_sample_loss = -negative_score.mean()
        else:
            positive_sample_loss = -(subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = -(subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        torch.distributed.all_reduce(loss, group=gpc.get_group(ParallelMode.SEQUENCE))

        return loss, positive_sample_loss, negative_sample_loss
