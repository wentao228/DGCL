import torch as t
import torch.nn.functional as F


def calcRegLoss(model):
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def contrastLoss(embeds1, embeds2, nodes, temp):
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    return -t.log(nume / deno).mean()


def ce(pred, target):
    return F.cross_entropy(pred, target)


def l2_norm(x):
    epsilon = t.FloatTensor([1e-12]).cuda()
    # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
    return x / (t.max(t.norm(x, dim=1, keepdim=True), epsilon))
