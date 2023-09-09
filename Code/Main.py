import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model_sparse import Model
from DataHandler import DataHandler
from Utils.Utils import *
import os
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import random
import wandb


# Function to set random seed for reproducibility
def set_seed(seed):
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('DRUG', args.drug, 'GENE', args.gene)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Acc']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    # Function to create a formatted print statement
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    # Function to perform external testing
    def external_test_run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        return reses['Acc']

    # Function to train and evaluate the model
    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            train_loss = reses
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                test_r = reses
                log(self.makePrint('Test', ep, reses, tstFlag))
            logs = {'loss_all': train_loss['Loss'], 'loss_pre': train_loss['preLoss'],
                    'test_acc': test_r['Acc']}
            wandb.log(logs)
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.save_model('{}'.format(config['iteration']))
        return reses['Acc']

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

    # Function to train a single epoch
    def trainEpoch(self):
        self.model.train()
        trnLoader = self.handler.trnLoader
        epLoss, epPreLoss = 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        for i, tem in enumerate(trnLoader):
            drugs, genes, labels = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            ceLoss, sslLoss = self.model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
            sslLoss = sslLoss * args.ssl_reg

            regLoss = calcRegLoss(self.model) * args.reg
            loss = ceLoss + regLoss + sslLoss
            epLoss += loss.item()
            epPreLoss += ceLoss.item()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        return ret

    # Function to test a single epoch
    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        i = 0
        for tem in tstLoader:
            i += 1
            drugs, genes, labels = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            pre = self.model.predict(self.handler.torchBiAdj, drugs, genes)

            pre = F.log_softmax(pre, dim=1)
            pre = pre.data.max(1, keepdim=True)[1].detach().cpu()
            labels = labels.detach().cpu()
            epAcc = accuracy_score(labels, pre)
        ret = dict()
        ret['Acc'] = epAcc
        return ret

    # Function to load a pre-trained model
    def loadModel(self):
        self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

    # Function to save the trained model
    def save_model(self, model_path):
        model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        if not os.path.exists(model_parent_path):
            os.mkdir(model_parent_path)
        t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))


# Main execution block
if __name__ == '__main__':
    if args.is_debug is True:
        print("DEBUGGING MODE - Start without wandb")
        wandb.init(mode="disabled")
    else:
        wandb.init(project='HC', config=args)
        wandb.run.log_code(".")

    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
    if use_cuda:
        t.cuda.set_device(device)
    args.device = device

    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    config = dict()
    results = list()
    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result = coach.run()
        results.append(result)

    avg_r = np.mean(np.array(results), axis=0)
    std_r = np.std(results, axis=0)
    print('test results: ')
    print(avg_r)
    print(std_r)

    results.append(avg_r)
    results.append(std_r)

    results_parent_path = os.path.join(wandb.run.dir, 'results')
    if not os.path.exists(results_parent_path):
        os.mkdir(results_parent_path)
    np.savetxt('{}/{}_result.txt'.format(results_parent_path, args.data), np.array(results), delimiter=",", fmt='%f')

    print('result saved!!!')
    wandb.finish()
