from config_multi_desc import *
from model_multi_desc import *
from dataset import DataSet 
from logger_multi_desc import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm

from module.gcn.st_gcn import Model

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

seed = os.environ.get('MANUAL_SEED', 0)
seed = int(seed)
print('Set seed: %d' % seed)
setup_seed(seed)

# %%
class Processor:

    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label, batch_size, language_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
        
        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = F.normalize(self.full_language, dim=-1)
        self.full_language = self.full_language.cuda()
        assert len(self.full_language.shape) == 3, "language shape must be 3 dim"
        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict)

    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        else:
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def layernorm(self, feature):

        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature-mean) / torch.sqrt(var)

        return out

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, visual_size, language_size, weight_path,
                    training_strategy, training_strategy_args,
                    testing_strategy, testing_strategy_args):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.model = MI(visual_size, language_size,
                        training_strategy, training_strategy_args,
                        testing_strategy, testing_strategy_args).cuda()
        self.load_weights(self.encoder, weight_path)

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.model.parameters()}],
             lr=lr,
             weight_decay=weight_decay,
             )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

    @ex.capture
    def optimize(self, epoch_num, save_all_param, save_path, _log):
        _log.info("main track")
        for epoch in range(epoch_num):
            self.train_epoch_v2(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            _log.info("epoch [{}] dim loss: {}".format(epoch,self.dim_loss))
            if save_all_param:
                self.save_model(save_path=save_path.replace('_train.pt', '_epoch{}.pt'.format(epoch)))
            if epoch > 15:
                _log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
                _log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            else:
                _log.info("epoch [{}] : warm up epoch.".format(epoch))

    @ex.capture
    def train_epoch_v2(self, epoch, lr, margin, dataset, unseen_label, sub_loss_margin, sub_loss_weight,
                       temp_mask, max_remove_joint, remove_joint):
        self.encoder.eval()
        self.model.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr)
        running_loss = []
        loader = self.data_loader['train']
        if dataset == 'ntu60':
            seen_label = [i for i in range(60) if i not in unseen_label]
            label_mapping = -1 * torch.ones((60, )).long().cuda()
        elif dataset == 'ntu120':
            seen_label = [i for i in range(120) if i not in unseen_label]
            label_mapping = -1 * torch.ones((120, )).long().cuda()
        elif dataset == 'pkummd':
            seen_label = [i for i in range(51) if i not in unseen_label]
            label_mapping = -1 * torch.ones((51, )).long().cuda()
        else:
            raise ValueError("unknown dataset: %s" % dataset)
        seen_label = torch.Tensor(seen_label).long().cuda()
        seen_lang = self.full_language[seen_label] # n_seen, n_desc, n_dim
        label_mapping[seen_label] = torch.arange(len(seen_label)).long().cuda()

        self.model.cur_ep = epoch
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()

            # no neg samples if batch_size == 1
            if data.shape[0] == 1:
                continue

            input0 = data.clone()
            feat0 = self.encoder(input0).detach()
            mapped_label = label_mapping[label]

            sub_visual = None

            loss = self.model.forward_v2(feat0, seen_lang, mapped_label, sub_visual=sub_visual,
                                         sub_loss_margin=sub_loss_margin, sub_loss_weight=sub_loss_weight)

            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def test_epoch(self, unseen_label, epoch, save_path):
        self.encoder.eval()
        self.model.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        acc_list = []
        for data, label in tqdm(loader):

            # y_t = label.numpy().tolist()
            # y_true += y_t

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[unseen_label]
            # inference
            feature = self.encoder(data)
            acc_batch, pred = self.model.get_acc(feature, unseen_language, label)

            # y_p = pred.cpu().numpy().tolist()
            # y_pred += y_p


            acc_list.append(acc_batch)
        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if epoch>15 and acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            self.save_model(save_path=save_path.replace('_train.pt', '_best.pt'))
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # np.save("y_true_3.npy",y_true)
            # np.save("y_pred_3.npy",y_pred)
            # print("save ok!")
        self.test_acc = acc

    @ex.capture
    def test_model(self, unseen_label, work_dir, save_name, result_prefix=''):
        self.encoder.eval()
        self.model.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        pred_logit = []
        acc_list = []
        for data, label in tqdm(loader):

            y_t = label.numpy().tolist()
            y_true += y_t

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[unseen_label]
            # inference
            feature = self.encoder(data)
            acc_batch, pred, logit = self.model.get_acc(feature, unseen_language, label, return_logit=True)

            y_p = pred.cpu().numpy().tolist()
            y_pred += y_p

            acc_list.append(acc_batch)
            pred_logit.append(logit.cpu().numpy())
        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()

        save_path = os.path.join(work_dir, save_name, 'result')
        os.makedirs(save_path, exist_ok=True)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        pred_logit = np.concatenate(pred_logit, axis=0)
        np.save(os.path.join(save_path, result_prefix + "y_label.npy"), y_true)
        np.save(os.path.join(save_path, result_prefix + "y_pred.npy"), y_pred)
        np.save(os.path.join(save_path, result_prefix + "y_logit.npy"), pred_logit)
        print("prediction result is saved to %s" % save_path)
        self.test_acc = acc

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        torch.save(self.model, save_path)

    def start(self):
        self.initialize()
        self.optimize()
        self.save_model()

    @ex.capture
    def test(self, save_path):
        self.load_data()
        self.load_model()
        
        if '_train.pt' in save_path and os.path.exists(save_path.replace('_train.pt', '_best.pt')):
            print('best acc weight detected')
            print('load weights from {}'.format(save_path.replace('_train.pt', '_best.pt')))
            # self.load_weights(self.model, save_path)
            self.model = torch.load(save_path.replace('_train.pt', '_best.pt')).cuda()
            with torch.no_grad():
                self.test_model(result_prefix='best_')
            print("test acc: {}".format(self.test_acc))

        print('load weights from {}'.format(save_path))
        # self.load_weights(self.model, save_path)
        self.model = torch.load(save_path).cuda()
        with torch.no_grad():
            self.test_model(result_prefix='last_')
        print("test acc: {}".format(self.test_acc))

class SotaProcessor:

    @ex.capture
    def load_data(self, sota_train_list, sota_train_label, 
        sota_test_list, sota_test_label, batch_size, language_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
         
        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = F.normalize(self.full_language,dim=-1)
        self.full_language = self.full_language.cuda()
        assert len(self.full_language.shape) == 3, "language shape must be 3 dim"

        self.dataset['train'] = DataSet(sota_train_list, sota_train_label)
        self.dataset['test'] = DataSet(sota_test_list, sota_test_label)

        # data seed tune
        rand_generator = torch.Generator()
        rand_generator.manual_seed(int(os.environ.get('MANUAL_DATA_SEED', 0)))
        print('Data seed: %d'%int(os.environ.get('MANUAL_DATA_SEED', 0)))
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            # shuffle=True,
            sampler=torch.utils.data.RandomSampler(self.dataset['train'], replacement=True, generator=rand_generator),)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15):

            if current_epoch < warmup_epoch:
                lr = lr_max * current_epoch / warmup_epoch
            else:
                lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, visual_size, language_size, weight_path):
        self.model = MI(visual_size, language_size).cuda()

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}],
             lr=lr,
             weight_decay=weight_decay,
             )

    @ex.capture
    def optimize(self, epoch_num, _log, save_path):
        _log.info("sota track")
        for epoch in range(epoch_num):
            self.train_epoch_v2(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            _log.info("epoch [{}] dim loss: {}".format(epoch,self.dim_loss))
            _log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            _log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            torch.save(self.model, save_path)

    @ex.capture
    def train_epoch(self, epoch, lr):
        self.model.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr)
        running_loss = []
        loader = self.data_loader['train']
        self.model.cur_ep = epoch
        for data, label in tqdm(loader):

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            seen_language = self.full_language[label]
            
            # Global
            feat0 = data.clone()
            dim = self.model(feat0, seen_language)

            # Loss
            loss = -dim
            
            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def train_epoch_v2(self, epoch, lr, margin, dataset, unseen_label, sub_loss_margin, sub_loss_weight,
                       temp_mask, max_remove_joint, remove_joint, neg_samples):
        self.model.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr)
        running_loss = []
        loader = self.data_loader['train']
        if dataset == 'ntu60':
            seen_label = [i for i in range(60) if i not in unseen_label]
            label_mapping = -1 * torch.ones((60, )).long().cuda()
        elif dataset == 'ntu120':
            seen_label = [i for i in range(120) if i not in unseen_label]
            label_mapping = -1 * torch.ones((120, )).long().cuda()
        elif dataset == 'pkummd':
            seen_label = [i for i in range(51) if i not in unseen_label]
            label_mapping = -1 * torch.ones((51, )).long().cuda()
        else:
            raise ValueError("unknown dataset: %s" % dataset)
        seen_label = torch.Tensor(seen_label).long().cuda()
        seen_lang = self.full_language[seen_label] # n_seen, n_desc, n_dim
        label_mapping[seen_label] = torch.arange(len(seen_label)).long().cuda()

        self.model.cur_ep = epoch
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()

            # no neg samples if batch_size == 1
            if data.shape[0] == 1:
                continue

            feat0 = data.clone()
            mapped_label = label_mapping[label]

            sub_visual = None

            loss = self.model.forward_v2(feat0, seen_lang, mapped_label, sub_visual=sub_visual,
                                         sub_loss_margin=sub_loss_margin, sub_loss_weight=sub_loss_weight,
                                         neg_samples=neg_samples)

            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def test_epoch(self, sota_unseen, epoch, save_path):
        self.model.eval()

        total = 0
        correct = 0
        loader = self.data_loader['test']
        acc_list = []
        for data, label in tqdm(loader):
            feature = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[sota_unseen]
            # inference
            acc_batch, pred = self.model.get_acc(feature, unseen_language, label, sota_unseen)
            acc_list.append(acc_batch)
        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            torch.save(self.model, save_path.replace('_train.pt', '_best.pt'))
        self.test_acc = acc



    @ex.capture
    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    def start(self):
        self.initialize()
        self.optimize()

    @ex.capture
    def test(self, save_path):
        self.load_data()
        self.load_model()
        
        if os.path.exists(save_path.replace('_train.pt', '_best.pt')):
            print('best acc weight detected')
            print('load weights from {}'.format(save_path.replace('_train.pt', '_best.pt')))
            # self.load_weights(self.model, save_path)
            self.model = torch.load(save_path.replace('_train.pt', '_best.pt')).cuda()
            with torch.no_grad():
                self.test_model(result_prefix='best_')
            print("test acc: {}".format(self.test_acc))

        # print('load weights from {}'.format(save_path))
        # # self.load_weights(self.model, save_path)
        # self.model = torch.load(save_path).cuda()
        # with torch.no_grad():
        #     self.test_model(result_prefix='last_')
        # print("test acc: {}".format(self.test_acc))

    @ex.capture
    def test_model(self, sota_unseen, work_dir, save_name, result_prefix=''):
        self.model.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        pred_logit = []
        acc_list = []
        for data, label in tqdm(loader):
            y_t = label.numpy().tolist()
            y_true += y_t

            feature = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            unseen_language = self.full_language[sota_unseen]
            # inference
            acc_batch, pred, logit = self.model.get_acc(feature, unseen_language, label, sota_unseen, return_logit=True)

            y_p = pred.cpu().numpy().tolist()
            y_pred += y_p

            acc_list.append(acc_batch)
            pred_logit.append(logit.cpu().numpy())
        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()

        save_path = os.path.join(work_dir, save_name, 'result')
        os.makedirs(save_path, exist_ok=True)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        pred_logit = np.concatenate(pred_logit, axis=0)
        np.save(os.path.join(save_path, result_prefix + "y_label.npy"), y_true)
        np.save(os.path.join(save_path, result_prefix + "y_pred.npy"), y_pred)
        np.save(os.path.join(save_path, result_prefix + "y_logit.npy"), pred_logit)
        print("prediction result is saved to %s" % save_path)
        self.test_acc = acc

# %%
@ex.automain
def main(track, phrase, _log, log_path, save_path, seed):
    # make dirs
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if "sota" in track:
        _log.info('select sota track')
        p = SotaProcessor()
    elif "main" in track:
        _log.info('select main track')
        p = Processor()

    if phrase == 'train':
        p.start()
    elif phrase == 'test':
        p.test()
