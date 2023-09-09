import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import pandas as pd
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader


class DataHandler:
    def __init__(self):
        self.data = args.data

    def map_data(self, data):
        """
        Map data to proper indices in case they are not in a continues [0, N) range
        Parameters
        ----------
        data : np.int32 arrays
        Returns
        -------
        mapped_data : np.int32 arrays
        n : length of mapped_data
        """
        uniq = list(set(data))

        id_dict = {old: new for new, old in enumerate(sorted(uniq))}
        data = np.array([id_dict[x] for x in data])
        n = len(uniq)

        return data, id_dict, n

    def load_data_from_database(self, dataset, mode='transductive', testing=True, relation_map=None,
                                post_relation_map=None):
        """
        Loads official train/test split and uses 10% of training samples for validaiton
        For each split computes 1-of-num_classes labels. Also computes training
        adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
        """

        dtypes = {
            'd_nodes': np.str, 'g_nodes': np.int32,
            'relations': np.float32}

        filename_train = '../Data/' + dataset + '/' + mode + '/train.csv'
        filename_test = '../Data/' + dataset + '/' + mode + '/test.csv'

        data_train = pd.read_csv(
            filename_train, header=None,
            names=['d_nodes', 'g_nodes', 'relations'], dtype=dtypes)

        data_test = pd.read_csv(
            filename_test, header=None,
            names=['d_nodes', 'g_nodes', 'relations'], dtype=dtypes)

        data_array_train = data_train.values.tolist()
        data_array_train = np.array(data_array_train)
        data_array_test = data_test.values.tolist()
        data_array_test = np.array(data_array_test)

        data_array = np.concatenate([data_array_train, data_array_test], axis=0)

        d_nodes_relations = data_array[:, 0].astype(dtypes['d_nodes'])
        g_nodes_relations = data_array[:, 1].astype(dtypes['g_nodes'])

        relations = data_array[:, 2].astype(dtypes['relations'])
        if relation_map is not None:
            for i, x in enumerate(relations):
                relations[i] = relation_map[x]

        d_nodes_relations, d_dict, num_drugs = self.map_data(d_nodes_relations)
        g_nodes_relations, g_dict, num_genes = self.map_data(g_nodes_relations)

        d_nodes_relations, g_nodes_relations = d_nodes_relations.astype(np.int64), g_nodes_relations.astype(np.int32)
        relations = relations.astype(np.float64)

        d_nodes = d_nodes_relations
        g_nodes = g_nodes_relations

        neutral_relation = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1
        # assumes that relations_train contains at least one example of every relation type
        relation_dict = {r: i for i, r in enumerate(np.sort(np.unique(relations)).tolist())}

        labels = np.full((num_drugs, num_genes), neutral_relation, dtype=np.int32)
        labels[d_nodes, g_nodes] = np.array([relation_dict[r] for r in relations])

        for i in range(len(d_nodes)):
            assert (labels[d_nodes[i], g_nodes[i]] == relation_dict[relations[i]])

        labels = labels.reshape([-1])

        # number of test and validation edges, see cf-nade code
        num_train = data_array_train.shape[0]
        num_test = data_array_test.shape[0]
        num_val = int(np.ceil(num_train * 0.2))
        num_train = num_train - num_val

        pairs_nonzero = np.array([[d, g] for d, g in zip(d_nodes, g_nodes)])
        idx_nonzero = np.array([d * num_genes + g for d, g in pairs_nonzero])

        for i in range(len(relations)):
            assert (labels[idx_nonzero[i]] == relation_dict[relations[i]])

        idx_nonzero_train = idx_nonzero[0:num_train + num_val]
        idx_nonzero_test = idx_nonzero[num_train + num_val:]

        pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
        pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

        # Internally shuffle training set (before splitting off validation set)
        rand_idx = list(range(len(idx_nonzero_train)))
        np.random.seed(42)
        np.random.shuffle(rand_idx)
        idx_nonzero_train = idx_nonzero_train[rand_idx]
        pairs_nonzero_train = pairs_nonzero_train[rand_idx]

        idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
        pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

        val_idx = idx_nonzero[0:num_val]
        train_idx = idx_nonzero[num_val:num_train + num_val]
        test_idx = idx_nonzero[num_train + num_val:]

        assert (len(test_idx) == num_test)

        val_pairs_idx = pairs_nonzero[0:num_val]
        train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
        test_pairs_idx = pairs_nonzero[num_train + num_val:num_train + num_val + num_test]

        d_test_idx, g_test_idx = test_pairs_idx.transpose()
        d_val_idx, g_val_idx = val_pairs_idx.transpose()
        d_train_idx, g_train_idx = train_pairs_idx.transpose()

        # create labels
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        if not args.validate:
            d_train_idx = np.hstack([d_train_idx, d_val_idx])
            g_train_idx = np.hstack([g_train_idx, g_val_idx])
            train_labels = np.hstack([train_labels, val_labels])
            # for adjacency matrix construction

            train_idx = np.hstack([train_idx, val_idx])

        class_values = np.sort(np.unique(relations))

        # make training adjacency matrix
        relation_mx_train = np.zeros(num_drugs * num_genes, dtype=np.float32)
        relation_mx_test = np.zeros(num_drugs * num_genes, dtype=np.float32)
        if post_relation_map is None:
            relation_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
            relation_mx_test[test_idx] = labels[test_idx].astype(np.float32) + 1.
        else:
            relation_mx_train[train_idx] = np.array(
                [post_relation_map[r] for r in class_values[labels[train_idx]]]) + 1.

        relation_mx_train = sp.csr_matrix(relation_mx_train.reshape(num_drugs, num_genes))
        relation_mx_test = sp.csr_matrix(relation_mx_test.reshape(num_drugs, num_genes))

        # make external testing set
        if dataset == 'LINCS':
            filename_external_test = '../Data/' + dataset + '/' + mode + '/external_test.csv'
            data_external_test = pd.read_csv(
                filename_external_test, header=None,
                names=['d_nodes', 'g_nodes', 'relations'], dtype=dtypes)
            data_array_external_test = data_external_test.values.tolist()
            data_array_external_test = np.array(data_array_external_test)

            d_nodes_external_relations = data_array_external_test[:, 0].astype(dtypes['d_nodes'])
            g_nodes_external_relations = data_array_external_test[:, 1].astype(dtypes['g_nodes'])

            external_test_relations = data_array_external_test[:, 2].astype(dtypes['relations'])

            external_test_relations = external_test_relations.astype(np.float64)

            d_external_test_nodes = d_nodes_external_relations
            g_external_test_nodes = g_nodes_external_relations

            d_external_test_idx = np.array([d_dict[d] for d in d_external_test_nodes])

            g_external_test_idx = np.array([g_dict[g] for g in g_external_test_nodes])

            external_test_labels = np.array([relation_dict[r] for r in external_test_relations])

            d_test_idx, g_test_idx, test_labels = d_external_test_idx, g_external_test_idx, external_test_labels

        return relation_mx_train, relation_mx_test, train_labels, d_train_idx, g_train_idx, \
            val_labels, d_val_idx, g_val_idx, test_labels, d_test_idx, g_test_idx, class_values

    def normalizeAdj(self, mat):
        """
        Normalize an adjacency matrix using the degree normalization technique.

        Parameters:
        mat (sparse matrix): The input adjacency matrix to be normalized.

        Returns:
        sparse matrix: The normalized adjacency matrix.
        """
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        """
        Convert a SciPy sparse matrix into a PyTorch sparse tensor and apply normalization.

        Parameters:
        mat (sparse matrix): The input sparse matrix to be converted and normalized.

        Returns:
        torch.sparse.FloatTensor: A PyTorch sparse tensor with applied normalization.
        """
        a = sp.csr_matrix((args.drug, args.drug))
        b = sp.csr_matrix((args.gene, args.gene))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def LoadData(self):
        """
        This method loads the dataset, preprocesses it, and creates data loaders for training and testing.
        """
        relation_mx_train, relation_mx_test, train_labels, d_train_idx, g_train_idx, \
            val_labels, d_val_idx, g_val_idx, test_labels, d_test_idx, g_test_idx, class_values = self.load_data_from_database(
            args.data)

        # Apply thresholding to the adjacency matrices
        trnMat, tstMat = relation_mx_train, relation_mx_test
        trnMat[trnMat >= 1] = 1
        tstMat[tstMat >= 1] = 1

        if type(trnMat) != coo_matrix:
            trnMat = sp.coo_matrix(trnMat)
        if type(tstMat) != coo_matrix:
            tstMat = sp.coo_matrix(tstMat)
        args.drug, args.gene = trnMat.shape
        args.num_classes = len(class_values)
        self.torchBiAdj = self.makeTorchAdj(trnMat)

        trnData = TrnData(train_labels, d_train_idx, g_train_idx)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=False,
                                               num_workers=0)  # already shuffled training set
        if args.validate:
            tstData = TstData(val_labels, d_val_idx, g_val_idx)
        else:
            tstData = TstData(test_labels, d_test_idx, g_test_idx)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)


# Data loader for training data
class TrnData(data.Dataset):
    def __init__(self, train_labels, d_train_idx, g_train_idx):
        self.train_labels = train_labels
        self.d_train_idx = d_train_idx
        self.g_train_idx = g_train_idx

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return self.d_train_idx[idx], self.g_train_idx[idx], self.train_labels[idx]


# Data loader for testing data
class TstData(data.Dataset):
    def __init__(self, test_labels, d_test_idx, g_test_idx):
        self.test_labels = test_labels
        self.d_test_idx = d_test_idx
        self.g_test_idx = g_test_idx

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return self.d_test_idx[idx], self.g_test_idx[idx], self.test_labels[idx]
