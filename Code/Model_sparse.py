import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Initialize drug and gene embeddings
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))

        # Initialize GCN (Graph Convolutional Network) layer
        self.gcnLayer = GCNLayer()

        # Initialize HGNN (Hypergraph Neural Network) layer
        self.hgnnLayer = HGNNLayer()

        # Initialize classifier layer
        self.classifierLayer = ClassifierLayer()

        # Initialize trainable parametric matrices for drug-hyperedge matrix and gene-hyperedge matrix
        if args.dense:
            self.dHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
            self.gHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

        # Initialize SpAdjDropEdge layer for dropout on the graph edges
        self.edgeDropper = SpAdjDropEdge()

    def forward(self, adj, keepRate):
        # Concatenate drug and gene embeddings
        embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        hyperEmbedsLst = [embeds]

        # approximate the drug-hyperedge matrix and gene-hyperedge matrix
        ddHyper = self.dEmbeds * args.mult
        ggHyper = self.gEmbeds * args.mult

        if args.dense:
            ddHyper = self.dEmbeds @ self.dHyper
            ggHyper = self.gEmbeds @ self.gHyper

        for i in range(args.gnn_layer):
            # Perform GCN layer operation
            gcnEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), embedsLst[-1])

            # Perform HGNN layer operation
            hyperDEmbeds = self.hgnnLayer(ddHyper, embedsLst[-1][:args.drug])
            hyperGEmbeds = self.hgnnLayer(ggHyper, embedsLst[-1][args.drug:])
            hyperEmbeds = t.concat([hyperDEmbeds, hyperGEmbeds], axis=0)

            # Append embeddings to lists
            gcnEmbedsLst.append(gcnEmbeds)
            hyperEmbedsLst.append(hyperEmbeds)
            embedsLst.append(gcnEmbeds + hyperEmbeds)

        # Sum all embeddings
        embeds = sum(embedsLst)
        return embeds, gcnEmbedsLst, hyperEmbedsLst

    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Calculate Cross-Entropy loss
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)

        # Calculate Self-Supervised Learning (SSL) loss
        sslLoss = 0
        for i in range(1, args.gnn_layer + 1, 1):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:args.drug], embeds2[:args.drug], t.unique(drugs),
                                    args.temp) + contrastLoss(
                embeds1[args.drug:], embeds2[args.drug:], t.unique(genes), args.temp)

        return ceLoss, sslLoss

    def predict(self, adj, drugs, genes):
        embeds, _, _ = self.forward(adj, 1.0)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Perform classification
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        return pre


# Define the GCN (Graph Convolutional Network) layer
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return l2_norm(t.spmm(adj, embeds))


# Define the HGNN (Hypergraph Neural Network) layer
class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()

    def forward(self, adj, embeds):
        lat = adj.T @ embeds
        ret = adj @ lat
        return l2_norm(ret)


# Define the SpAdjDropEdge layer for graph edge dropout
class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


# Define the ClassifierLayer for classification
class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(args.latdim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret
