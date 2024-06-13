from config_multi_desc import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat,rearrange
import random

from model.estimator import *

def init_weights(m):
    class_name=m.__class__.__name__

    if "Linear" in class_name:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
class GlobalDiscriminator(nn.Module):
    
    @ex.capture
    def __init__(self, in_feature):
        super().__init__()
        self.l0 = nn.Linear(in_feature, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)
        # self.apply(init_weights)

    def forward(self, visual, language):

        x = torch.cat((visual, language), dim=-1)
        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        out = self.l2(out)
        return out
    
class GlobalDiscriminatorFCResidual(nn.Module):
    
    @ex.capture
    def __init__(self, visual_size, language_size):
        super().__init__()
        self.proj = nn.Linear(visual_size, language_size)
        self.l0 = nn.Linear(3*language_size, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)
        self.ln = nn.LayerNorm([language_size],elementwise_affine=False)
        # self.apply(init_weights)

    def forward(self, visual, language):
        vis_proj = self.proj(visual)
        vis_proj = self.ln(vis_proj)

        x = torch.cat((language, vis_proj, torch.abs(language - vis_proj)), dim=-1)
        # x = torch.cat((language, vis_proj), dim=-1)
        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        out = self.l2(out)
        return out

class MI(nn.Module):

    @ex.capture
    def __init__(self, visual_size, language_size,
                 training_strategy, training_strategy_args,
                 testing_strategy, testing_strategy_args,
                 global_discriminator_arch):
        super(MI,self).__init__()

        if global_discriminator_arch == 'raw':
            self.global_D = GlobalDiscriminator(visual_size+language_size)
            self.ln = nn.LayerNorm([visual_size],elementwise_affine=False)
        elif global_discriminator_arch == 'fc_residual':
            self.global_D = GlobalDiscriminatorFCResidual(visual_size, language_size)
            self.ln = nn.Identity()

        self.training_strategy = training_strategy
        self.training_strategy_args = training_strategy_args
        self.testing_strategy = testing_strategy
        self.testing_strategy_args = testing_strategy_args

        self.cur_ep = 0
        self.loss_type = 'JSD'
        self.loss_kwargs = {}

        if 'attention' in self.training_strategy:
            if self.training_strategy_args['estimator_type'] == 'raw':
                network_constructor = SymmetricEstimator
            elif self.training_strategy_args['estimator_type'] == 'res':
                network_constructor = SymmetricResEstimator
            elif self.training_strategy_args['estimator_type'] == 'raw_silu':
                network_constructor = SymmetricEstimator_silu
            elif self.training_strategy_args['estimator_type'] == 'simple_parallel':
                network_constructor = SimpleParallelEstimator
            else:
                pass
            self.global_D = network_constructor(language_size)
            self.ln = nn.Identity()
            has_gate = self.training_strategy.startswith('gated')
            training_strategy = self.training_strategy.replace('gated_', '')
            if training_strategy.startswith('attention'):
                self.num_att = self.training_strategy_args['num_att']
                assert self.num_att > 0
                self.proj = nn.Sequential(*[nn.Linear(visual_size, language_size) for _ in range(self.num_att)])
                self.proj_ln = nn.Sequential(*[nn.LayerNorm([language_size], elementwise_affine=False) for _ in range(self.num_att)])
            elif training_strategy.startswith('mg_attention'):
                self.num_att = len(self.training_strategy_args['mg_att_series'])
                assert self.num_att > 0
                self.proj = nn.Sequential(*[nn.Linear(visual_size, language_size) for _ in range(self.num_att)])
                self.proj_ln = nn.Sequential(*[nn.LayerNorm([language_size], elementwise_affine=False) for _ in range(self.num_att)])
                if self.training_strategy_args['ortho_init']:
                    # choice1: unify orthogonal init (but not strictly orthogonal a@a.T != I)
                    if self.training_strategy_args['ortho_init'] == 'unify':
                        full_weight = torch.empty(language_size * self.num_att, visual_size).to(self.proj[0].weight.device)
                        nn.init.orthogonal_(full_weight)
                        for i in range(len(self.proj)):
                            self.proj[i].weight.data = full_weight[i * language_size: (i + 1) * language_size, :]
                    # choice2: orthogonal init for every projection (feature only orthogonal in projection space)
                    if self.training_strategy_args['ortho_init'] == 'each':   
                        for i in range(len(self.proj)):
                            nn.init.orthogonal_(self.proj[i].weight.data)
                if training_strategy.startswith('mg_attention_multi'):
                    self.global_D = nn.Sequential(
                        *[network_constructor(language_size) for _ in range(self.num_att)]
                    )
                if training_strategy == 'mg_attention_multi_ms':
                    pass
                    # self.extra_proj = nn.Sequential(*[nn.Linear(visual_size, language_size) for _ in range(self.num_att)])
                    # self.extra_proj_ln = nn.Sequential(*[nn.LayerNorm([language_size], elementwise_affine=False) for _ in range(self.num_att)])
            elif training_strategy == 'mg_attention_shared':
                self.num_att = len(self.training_strategy_args['mg_att_series'])
                assert self.num_att > 0
                self.proj = nn.Linear(visual_size, language_size)
                self.proj_ln = nn.LayerNorm([language_size], elementwise_affine=False)
            # initialization for MoE gating
            if has_gate:
                # input can be: visual only (v), visual language (vl)
                gate_type = self.training_strategy_args['gate_type']
                if gate_type == 'v':
                    # n, c -> n, num_expert
                    self.gate_fc = nn.Linear(visual_size, self.num_att)
                elif gate_type == 'vdeep':
                    # n, c -> n, num_expert
                    self.gate_fc = nn.Sequential(
                        nn.Linear(visual_size, 128),
                        nn.SiLU(),
                        nn.Linear(128, 64),
                        nn.SiLU(),
                        nn.Linear(64, self.num_att),
                        )
                elif gate_type == 'vl':
                    # n, n_cls, 2*c -> n, n_cls, num_expert, -> n, num_expert
                    self.gate_fc = nn.Linear(visual_size + language_size, self.num_att)
                elif gate_type == 'vldeep':
                    self.gate_fc = nn.Sequential(
                        nn.Linear(visual_size + language_size, 128),
                        nn.SiLU(),
                        nn.Linear(128, 64),
                        nn.SiLU(),
                        nn.Linear(64, self.num_att),
                        )
                elif gate_type == 'static':
                    self.gate_param = nn.Parameter(torch.ones((self.num_att, )).float(), requires_grad=True)

    def gate(self, visual, lang, topk=2):
        """
        Args:
            visual: (b, c)
            lang: (total_cls, n_desc, c)
            topk: int

            return: (b, num_expert)
        """
        gate_type = self.training_strategy_args['gate_type']
        if gate_type == 'static':
            g = self.gate_param.softmax(0)
            g = g.unsqueeze(0).repeat(visual.shape[0], 1)
        else:
            if 'vl' in gate_type:
                b, c = visual.shape
                total_cls, n_desc, _ = lang.shape
                visual = visual.unsqueeze(1).unsqueeze(1) # b, 1, 1, c
                lang = lang.unsqueeze(0) # 1, total_cls, n_desc, c
                g = self.gate_fc(torch.cat([visual.repeat(1, total_cls, n_desc, 1), lang.repeat(b, 1, 1, 1)], dim=-1)) # b, total_cls, n_desc, num_expert
                g = g.sum((1, 2)) # b, total_cls, n_desc, num_expert -> b, num_expert
                b_idx = torch.arange(visual.shape[0]).to(visual.device)
                idx = g.argsort(1, descending=True)
                g[b_idx.unsqueeze(-1), idx[:, topk:]] = -torch.inf
                g = g.softmax(1)
            else:
                # perform v only gating
                g = self.gate_fc(visual) # b, num_expert
                b_idx = torch.arange(visual.shape[0]).to(visual.device)
                idx = g.argsort(1, descending=True)
                g[b_idx.unsqueeze(-1), idx[:, topk:]] = -torch.inf
                g = g.softmax(1)
            
        return g

    def normfeat(self, x):
        return F.normalize(x, p=2, dim=-1)

    def att_lang_aggregation(self, visual, lang, agg_type='full', agg_kwargs={}):
        """
        Args:
            visual: (b, c)
            lang: (b, n_desc, c)
            agg_type: "full" or "topk", default is "full"
            agg_kwargs: dict of kwargs for aggregation

            return: (b, c)
        """
        rev_coeff = agg_kwargs['rev_coeff']
        att = rev_coeff * torch.bmm(lang, visual.unsqueeze(-1)) / torch.sqrt(torch.tensor(lang.shape[-1]).float().to(lang.device))
        if agg_type == 'topk':
            k = agg_kwargs['k']
            idx = torch.argsort(att, dim=1, descending=True).squeeze()[:, k - 1]
            att_thresh = att[torch.arange(att.shape[0]).to(att.device), idx]
            att = torch.masked_fill(att, att < att_thresh.unsqueeze(-1), -torch.inf)
        att = att.softmax(1)
        cur_lang = (att * lang).sum(1)
        return cur_lang
    
    def att_lang_aggregation_multi(self, visual, lang, agg_type='full', agg_kwargs={}):
        """
        Args:
            visual: (b, c)
            lang: (n_seen, n_desc, c)
            agg_type: "full" or "topk", default is "full"
            agg_kwargs: dict of kwargs for aggregation

            return: (b, n_seen, c)
        """
        rev_coeff = agg_kwargs.get('rev_coeff', 1)
        att = lang @ visual.transpose(1, 0) / torch.sqrt(torch.tensor(lang.shape[-1]).float().to(lang.device))
        att = rev_coeff * att
        if agg_type == 'topk':
            k = agg_kwargs['k']
            idx = torch.argsort(att, dim=1, descending=True).squeeze()[:, k - 1]
            if visual.shape[0] == 1:
                idx = idx.unsqueeze(-1)
            att_thresh = att[torch.arange(att.shape[0]).to(att.device).unsqueeze(-1).unsqueeze(-1), idx.unsqueeze(1), torch.arange(att.shape[-1]).to(att.device).unsqueeze(0).unsqueeze(0)]
            att = torch.masked_fill(att, att < att_thresh, -torch.inf)
        att = att.softmax(1)
        cur_lang = torch.bmm(att.permute((0, 2, 1)), lang) # n_seen, b, c_lang
        cur_lang = cur_lang.permute((1, 0, 2)) # b, n_seen, c_lang
        return cur_lang
       
    def info_nce(self, module, visual, language, label, sub_visual=None, act_func=F.softplus, neg_samples=8, 
                 sub_loss_margin=0.5, sub_loss_weight=0.2, sub_loss_type='v1', reduction='mean'):
        """
        Args:
            module: Torch Module
            visual: (b, c)
            language: (total_cls, c) or (b, total_cls, c)
            label: (b,) mapped to seen index
            sub_visual: (b, c) or None

            return: loss (1,)
        """
        b = visual.shape[0]
        if len(language.shape) == 2:
            seen_count = language.shape[0]
            visual_logit = act_func(module(visual.unsqueeze(1).repeat(1, seen_count, 1), language.unsqueeze(0).repeat(b, 1, 1)))
            if sub_visual is not None:
                sub_visual_logit = act_func(module(sub_visual.unsqueeze(1).repeat(1, seen_count, 1), language.unsqueeze(0).repeat(b, 1, 1)))
        elif len(language.shape) == 3:
            seen_count = language.shape[1]
            visual_logit = act_func(module(visual.unsqueeze(1).repeat(1, seen_count, 1), language))
            if sub_visual is not None:
                sub_visual_logit = act_func(module(sub_visual.unsqueeze(1).repeat(1, seen_count, 1), language))
        else:
            raise ValueError('Unknown language shape: %s' % str(language.shape))

        # x random sampling
        with torch.no_grad():
            positive_mask = torch.zeros_like(visual_logit).bool()
            b_idx = torch.arange(b).to(visual.device)
            positive_mask[b_idx, label] = 1
            
            max_batch_neg = b - positive_mask.sum(0).max() # max number of positive samples in batch, for x neg sampling
            cur_neg_sample = min(max_batch_neg, neg_samples)
            label_mask = label[None, :] == label[:, None]
            label_mask_cumsum = (~label_mask).cumsum(1)
            label_mask_cumsum[label_mask] = -1
            neg_idx = torch.randperm(label_mask_cumsum.max(1)[0].min()).to(visual.device).long()

            neg_samples_idx = neg_idx[:cur_neg_sample] + 1
            mapped_bool_samples_idx = (label_mask_cumsum[:, :, None] == neg_samples_idx[None, None, :]).sum(-1) # (b, b)
            mapped_samples_idx = mapped_bool_samples_idx.nonzero() # b*cur_neg_sample, 2
        pos_visual_logit = visual_logit[b_idx, label].unsqueeze(1)
        neg_visual_logit = visual_logit[mapped_samples_idx[:, 1], label[mapped_samples_idx[:, 0]]]
        neg_visual_logit = neg_visual_logit.reshape(b, cur_neg_sample, 1)
        cat_visual_logit = torch.cat([pos_visual_logit, neg_visual_logit], dim=1)

        cat_visual_logit = cat_visual_logit / 1 # temperature
        cat_visual_logit = F.log_softmax(cat_visual_logit, 1)
        loss1 = -cat_visual_logit[:, 0]

        if sub_visual is not None:
            pos_sub_visual_logit = sub_visual_logit[b_idx, label].unsqueeze(1)
            if sub_loss_type == 'v1':
                cat_sub_visual_logit = torch.cat([pos_sub_visual_logit, neg_visual_logit], dim=1)
                cat_sub_visual_logit = cat_sub_visual_logit / 1 # temperature
                cat_sub_visual_logit = F.log_softmax(cat_sub_visual_logit, 1)
                loss2 = -cat_sub_visual_logit[:, 0]
                loss = loss1 + sub_loss_weight*torch.clamp(sub_loss_margin - (-loss1 + loss2), min=0)
            elif sub_loss_type == 'v2':
                loss = loss1 + sub_loss_weight*torch.clamp(sub_loss_margin - (pos_visual_logit - pos_sub_visual_logit), min=0)
            elif sub_loss_type == 'v3':
                cat_visual_sub = torch.cat([pos_visual_logit, pos_sub_visual_logit], dim=1)
                cat_visual_sub = cat_visual_sub.softmax(1)
                loss = loss1 + sub_loss_weight*torch.clamp(sub_loss_margin - (cat_visual_sub[:, 0] - cat_visual_sub[:, 1]), min=0)
        else:
            loss = loss1

        if reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss
        return loss

    def forward_v2(self, visual, language, label, sub_visual=None, sub_loss_margin=0.5, sub_loss_weight=0.2, neg_samples=8):
        if self.training_strategy == 'mg_attention_multi_ms':
            # norm add
            desc_lang = language[:, :100, :] # (n_seen, n_desc, c)
            motion_lang = language[:, 100, :][:, None] # (n_seen, 1, c)
            # motion_lang = language[:, 100:, :]
            visual = self.ln(visual)
            if sub_visual is not None:
                sub_visual = self.ln(sub_visual)
            mg_att_series = self.training_strategy_args['mg_att_series']
            num_att = len(mg_att_series)
            dim = 0
            rev_coeff = -1 if self.cur_ep < self.training_strategy_args['reverse_ep'] else 1
            for i in range(num_att):
                k = mg_att_series[i]
                cur_visual = self.proj[i](visual)
                cur_visual = self.proj_ln[i](cur_visual)
                if sub_visual is not None:
                    cur_sub_visual = self.proj[i](sub_visual)
                    cur_sub_visual = self.proj_ln[i](cur_sub_visual)
                else:
                    cur_sub_visual = None
                if hasattr(self, 'extra_proj'):
                    extra_cur_visual = self.extra_proj[i](visual)
                    extra_cur_visual = self.extra_proj_ln[i](extra_cur_visual)
                # cur_desc_lang: (b, n_seen, c)
                cur_desc_lang = self.att_lang_aggregation_multi(cur_visual, desc_lang, agg_type='topk', agg_kwargs=dict(rev_coeff=rev_coeff, k=k))
                cur_motion_lang = motion_lang.squeeze(1).unsqueeze(0).repeat(cur_visual.shape[0], 1, 1) # for single motion
                # cur_motion_lang = self.att_lang_aggregation_multi(extra_cur_visual, motion_lang, agg_type='topk', agg_kwargs=dict(rev_coeff=rev_coeff, k=k))
                # cur_lang = torch.cat([cur_desc_lang, cur_motion_lang], dim=-1) # fuse tensor, need to be unpack
                cur_lang = cur_desc_lang + cur_motion_lang
                att_dim = self.info_nce(self.global_D[i], cur_visual, cur_lang, label, sub_visual=cur_sub_visual,
                                        sub_loss_margin=sub_loss_margin, sub_loss_weight=sub_loss_weight, neg_samples=neg_samples)
                dim = dim + att_dim
            dim = dim / num_att
        return dim

    @ex.capture
    def get_acc(self, visual, unseen_language, label, unseen_label, return_logit=False):
        
        bs = visual.shape[0]
        unsee_num = unseen_language.shape[0]
        num_desc = unseen_language.shape[1]

        if self.testing_strategy == 'mg_attention_multi_ms':
            unseen_desc_lang = unseen_language[:, :100, :] # (n_seen, n_desc, c)
            unseen_motion_lang = unseen_language[:, 100, :][:, None] # (n_seen, 1, c)
            # unseen_motion_lang = unseen_language[:, 100:, :]
            mg_att_series = self.testing_strategy_args['mg_att_series']
            num_att = len(mg_att_series)
            dim_list = 0
            for i in range(num_att):
                k = mg_att_series[i]
                cur_visual = self.proj[i](visual)
                cur_visual = self.proj_ln[i](cur_visual) # b, c_lang
                # extra visual
                if hasattr(self, 'extra_proj'):
                    extra_cur_visual = self.extra_proj[i](visual)
                    extra_cur_visual = self.extra_proj_ln[i](extra_cur_visual)
                # cur_desc_lang: (b, n_seen, c)
                cur_desc_lang = self.att_lang_aggregation_multi(cur_visual, unseen_desc_lang, agg_type='topk', agg_kwargs=dict(rev_coeff=1, k=k))
                cur_motion_lang = unseen_motion_lang.squeeze(1).unsqueeze(0).repeat(cur_visual.shape[0], 1, 1) # for single motion
                # cur_motion_lang = self.att_lang_aggregation_multi(extra_cur_visual, unseen_motion_lang, agg_type='topk', agg_kwargs=dict(rev_coeff=1, k=k))
                # cur_lang = torch.cat([cur_desc_lang, cur_motion_lang], dim=-1) # fuse tensor, need to be unpack
                cur_lang = cur_desc_lang + cur_motion_lang
                # cur_lang = torch.cat([cur_desc_lang, cur_motion_lang], dim=-1) # fuse tensor, need to be unpack
                cur_visual = repeat(cur_visual,'b c -> b u c',u = unsee_num) # b, un_seen, c_lang
                cur_dim_list = -F.softplus(-self.global_D[i](cur_visual, cur_lang)).squeeze(-1)
                dim_list = cur_dim_list + dim_list
            _, pred = torch.max(dim_list, 1)
        else:
            raise ValueError('Unkonwn testing strategy: {}'.format(self.testing_strategy))

        unseen_label = torch.tensor(unseen_label).cuda()
        pred = torch.index_select(unseen_label,0,pred)
        acc = pred.eq(label.view_as(pred)).float().mean()

        if return_logit:
            return acc, pred, dim_list
        return acc, pred
    
    @ex.capture
    def get_acc_v2(self, visual, unseen_language, label, unseen_label, return_logit=False):
        
        bs = visual.shape[0]
        unsee_num = unseen_language.shape[0]
        num_desc = unseen_language.shape[1]

        # post_func = lambda x: -F.softplus(-x) # JSD
        post_func = lambda x: F.silu(x) # InfoNCE

        if self.testing_strategy == 'sum':
            if 'attention' in self.training_strategy:
                if isinstance(self.proj, nn.Sequential):
                    if self.testing_strategy_args['projection'] == 'first':
                        visual = self.proj[0](visual)
                        visual = repeat(visual,'b c -> b u d c',u = unsee_num, d=num_desc)
                        unseen_language = repeat(unseen_language,'u d c -> b u d c', b = bs)
                    elif self.testing_strategy_args['projection'] == 'avg':
                        out_visual = 0
                        for i in range(len(self.proj)):
                            out_visual = out_visual + self.proj[i](visual)
                        visual = out_visual / len(self.proj)
                        visual = repeat(visual,'b c -> b u d c',u = unsee_num, d=num_desc)
                        unseen_language = repeat(unseen_language,'u d c -> b u d c', b = bs)
                    elif self.testing_strategy_args['projection'] == 'all':
                        out_visual = []
                        for i in range(len(self.proj)):
                            out_visual.append(self.proj[i](visual))
                        visual = torch.stack(out_visual, dim=1)
                        visual = repeat(visual,'b p c -> b u p d c',u = unsee_num, d=num_desc)
                        unseen_language = repeat(unseen_language,'u d c -> b u p d c', b = bs, p=len(self.proj))
                else:
                    visual = self.proj(visual)
                    visual = repeat(visual,'b c -> b u d c',u = unsee_num, d=num_desc)
                    unseen_language = repeat(unseen_language,'u d c -> b u d c', b = bs)
            else:
                visual = repeat(visual,'b c -> b u d c',u = unsee_num, d=num_desc)
                unseen_language = repeat(unseen_language,'u d c -> b u d c', b = bs)
            dim_list = post_func(self.global_D(visual, unseen_language)).squeeze(-1) # b, u, d
            if len(dim_list.shape) == 3:
                _, pred = torch.max(dim_list.sum(-1), 1)
            elif len(dim_list.shape) == 4:
                _, pred = torch.max(dim_list.sum(-1).sum(-1), 1)
            else:
                raise NotImplementedError
        elif self.testing_strategy == 'nearest':
            if 'attention' in self.training_strategy:
                if isinstance(self.proj, nn.Sequential):
                    if self.testing_strategy_args['projection'] == 'first':
                        visual = self.proj[0](visual)
                    elif self.testing_strategy_args['projection'] == 'avg':
                        out_visual = 0
                        for i in range(len(self.proj)):
                            out_visual = out_visual + self.proj[i](visual)
                        visual = out_visual / len(self.proj)
                else:
                    visual = self.proj(visual)
            visual = repeat(visual,'b c -> b u d c',u = unsee_num, d=num_desc)
            unseen_language = repeat(unseen_language,'u d c -> b u d c', b = bs)
            dim_list = post_func(self.global_D(visual, unseen_language)).squeeze(-1)
            _, pred = torch.max(dim_list.max(-1)[0], 1)
        elif self.testing_strategy == 'topk_vote':
            if 'attention' in self.training_strategy:
                if isinstance(self.proj, nn.Sequential):
                    if self.testing_strategy_args['projection'] == 'first':
                        visual = self.proj[0](visual)
                    elif self.testing_strategy_args['projection'] == 'avg':
                        out_visual = 0
                        for i in range(len(self.proj)):
                            out_visual = out_visual + self.proj[i](visual)
                        visual = out_visual / len(self.proj)
                else:
                    visual = self.proj(visual)
            k = self.testing_strategy_args['k']
            visual = repeat(visual,'b c -> b u d c',u = unsee_num, d=num_desc)
            unseen_language = repeat(unseen_language,'u d c -> b u d c', b = bs)
            dim_list = post_func(self.global_D(visual, unseen_language)).squeeze(-1)
            dim_idx = torch.arange(unsee_num).to(visual.device).unsqueeze(-1).repeat(1, num_desc).flatten()
            topk_idx = dim_list.reshape(bs, -1).argsort(-1, descending=True)[:, :k]
            map_cls_idx = dim_idx[topk_idx]
            pred = []
            for i in range(bs):
                uniq, count = map_cls_idx[i].unique(return_counts=True)
                if count.max() > 1:
                    pred.append(uniq[count.argmax()])
                else:
                    pred.append(map_cls_idx[i][0])
            pred = torch.Tensor(pred).to(visual.device).to(torch.int32)
        elif self.testing_strategy == 'avg_feat':
            if 'attention' in self.training_strategy:
                if isinstance(self.proj, nn.Sequential):
                    if self.testing_strategy_args['projection'] == 'first':
                        visual = self.proj[0](visual)
                    elif self.testing_strategy_args['projection'] == 'avg':
                        out_visual = 0
                        for i in range(len(self.proj)):
                            out_visual = out_visual + self.proj[i](visual)
                        visual = out_visual / len(self.proj)
                else:
                    visual = self.proj(visual)
            visual = repeat(visual,'b c -> b u c',u = unsee_num)
            unseen_language = repeat(unseen_language.mean(1),'u c -> b u c', b = bs)
            dim_list = post_func(self.global_D(visual, unseen_language)).squeeze(-1)
            _, pred = torch.max(dim_list, 1)
        elif self.testing_strategy == 'attention':
            num_att = self.training_strategy_args['num_att']
            dim_list = 0
            for i in range(num_att):
                cur_visual = self.proj[i](visual)
                cur_visual = self.proj_ln[i](cur_visual) # b, c_lang
                att = unseen_language @ cur_visual.transpose(1, 0) / torch.sqrt(torch.tensor(unseen_language.shape[-1]).float().to(unseen_language.device))
                att = att.softmax(1) # n_unseen, n_desc, b
                cur_lang = torch.bmm(att.permute((0, 2, 1)), unseen_language) # n_unseen, b, c_lang
                cur_lang = cur_lang.permute((1, 0, 2)) # b, un_seen, c_lang
                cur_visual = repeat(cur_visual,'b c -> b u c',u = unsee_num) # b, un_seen, c_lang
                cur_dim_list = post_func(self.global_D(cur_visual, cur_lang)).squeeze(-1)
                dim_list = cur_dim_list + dim_list
            _, pred = torch.max(dim_list, 1)
        elif self.testing_strategy == 'attention_topk':
            num_att = self.training_strategy_args['num_att']
            k = self.training_strategy_args['k']
            dim_list = 0
            for i in range(num_att):
                cur_visual = self.proj[i](visual)
                cur_visual = self.proj_ln[i](cur_visual) # b, c_lang
                att = unseen_language @ cur_visual.transpose(1, 0) / torch.sqrt(torch.tensor(unseen_language.shape[-1]).float().to(unseen_language.device))
                idx = torch.argsort(att, dim=1, descending=True).squeeze()[:, k - 1]
                att_thresh = att[torch.arange(att.shape[0]).to(att.device).unsqueeze(-1).unsqueeze(-1), idx.unsqueeze(1), torch.arange(att.shape[-1]).to(att.device).unsqueeze(0).unsqueeze(0)]
                att = torch.masked_fill(att, att < att_thresh, -torch.inf)
                att = att.softmax(1)
                cur_lang = torch.bmm(att.permute((0, 2, 1)), unseen_language) # n_unseen, b, c_lang
                cur_lang = cur_lang.permute((1, 0, 2)) # b, un_seen, c_lang
                cur_visual = repeat(cur_visual,'b c -> b u c',u = unsee_num) # b, un_seen, c_lang
                cur_dim_list = post_func(self.global_D(cur_visual, cur_lang)).squeeze(-1)
                dim_list = cur_dim_list + dim_list
            _, pred = torch.max(dim_list, 1)
        elif self.testing_strategy == 'mg_attention':
            mg_att_series = self.testing_strategy_args['mg_att_series']
            num_att = len(mg_att_series)
            dim_list = 0
            for i in range(num_att):
                k = mg_att_series[i]
                cur_visual = self.proj[i](visual)
                cur_visual = self.proj_ln[i](cur_visual) # b, c_lang
                att = unseen_language @ cur_visual.transpose(1, 0) / torch.sqrt(torch.tensor(unseen_language.shape[-1]).float().to(unseen_language.device))
                idx = torch.argsort(att, dim=1, descending=True).squeeze()[:, k - 1]
                att_thresh = att[torch.arange(att.shape[0]).to(att.device).unsqueeze(-1).unsqueeze(-1), idx.unsqueeze(1), torch.arange(att.shape[-1]).to(att.device).unsqueeze(0).unsqueeze(0)]
                att = torch.masked_fill(att, att < att_thresh, -torch.inf)
                # debug code
                # save_series = len(os.listdir('vis'))
                # torch.save(att.detach().cpu(), 'vis/%d_%d.pt'%(save_series // num_att, i))
                # debug code end
                att = att.softmax(1)
                cur_lang = torch.bmm(att.permute((0, 2, 1)), unseen_language) # n_unseen, b, c_lang
                cur_lang = cur_lang.permute((1, 0, 2)) # b, un_seen, c_lang
                cur_visual = repeat(cur_visual,'b c -> b u c',u = unsee_num) # b, un_seen, c_lang
                cur_dim_list = post_func(self.global_D(cur_visual, cur_lang)).squeeze(-1)
                dim_list = cur_dim_list + dim_list
            _, pred = torch.max(dim_list, 1)
        elif self.testing_strategy == 'mg_attention_shared':
            mg_att_series = self.testing_strategy_args['mg_att_series']
            num_att = len(mg_att_series)
            dim_list = 0
            for i in range(num_att):
                k = mg_att_series[i]
                cur_visual = self.proj(visual)
                cur_visual = self.proj_ln(cur_visual) # b, c_lang
                att = unseen_language @ cur_visual.transpose(1, 0) / torch.sqrt(torch.tensor(unseen_language.shape[-1]).float().to(unseen_language.device))
                idx = torch.argsort(att, dim=1, descending=True).squeeze()[:, k - 1]
                att_thresh = att[torch.arange(att.shape[0]).to(att.device).unsqueeze(-1).unsqueeze(-1), idx.unsqueeze(1), torch.arange(att.shape[-1]).to(att.device).unsqueeze(0).unsqueeze(0)]
                att = torch.masked_fill(att, att < att_thresh, -torch.inf)
                att = att.softmax(1)
                cur_lang = torch.bmm(att.permute((0, 2, 1)), unseen_language) # n_unseen, b, c_lang
                cur_lang = cur_lang.permute((1, 0, 2)) # b, un_seen, c_lang
                cur_visual = repeat(cur_visual,'b c -> b u c',u = unsee_num) # b, un_seen, c_lang
                cur_dim_list = post_func(self.global_D(cur_visual, cur_lang)).squeeze(-1)
                dim_list = cur_dim_list + dim_list
            _, pred = torch.max(dim_list, 1)
        else:
            raise ValueError('Unkonwn testing strategy: {}'.format(self.testing_strategy))

        unseen_label = torch.tensor(unseen_label).cuda()
        pred = torch.index_select(unseen_label,0,pred)
        acc = pred.eq(label.view_as(pred)).float().mean()

        if return_logit:
            return acc, pred, dim_list
        return acc, pred

@ex.capture
def temp_mask(data, mask_frame):
    x = data.clone()
    n, c, t, v, m = x.shape
    remain_num = t - mask_frame
    remain_frame = random.sample(range(t), remain_num)
    remain_frame.sort()
    x = x[:, :, remain_frame, :, :]

    return x
    
def motion_att_temp_mask(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion = torch.zeros_like(temp)
    motion[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
    motion = -(motion)**2
    temporal_att = motion.mean((1,3,4))

    ## The frames with the smallest att are reserved
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)
    temp_resample = temp.gather(2,temp_list)

    ## random temp mask
    random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    random_frame.sort()
    output = temp_resample[:, :, random_frame, :, :]

    return output
    
def motion_att_temp_mask2(data, mask_frame):

    n, c, t, v, m = data.shape
    temp = data.clone()
    remain_num = t - mask_frame

    ## get the motion_attention value
    motion_pre = torch.zeros_like(temp)
    motion_nex = torch.zeros_like(temp)
    motion_pre[:, :, :-1, :, :] = temp[:, :, 1:, :, :] - temp[:, :, :-1, :, :]
    motion_nex[:, :, 1:, :, :] = temp[:, :, :-1, :, :] - temp[:, :, 1:, :, :]
    motion = -((motion_pre)**2+(motion_nex)**2)
    temporal_att = motion.mean((1,3,4)) # n, c, t, v, m -> n, t

    ## The frames with the smallest att are reserved
    ## small att -> large motion
    _,temp_list = torch.topk(temporal_att, remain_num)
    temp_list,_ = torch.sort(temp_list.squeeze())
    temp_list = repeat(temp_list,'n t -> n c t v m',c=c,v=v,m=m)
    temp_resample = temp.gather(2,temp_list) # n, c, 35 frames, v, m

    ## random temp mask
    random_frame = random.sample(range(remain_num), remain_num-mask_frame)
    random_frame.sort()
    output = temp_resample[:, :, random_frame, :, :] # n, c, 20 frames, v, m

    return output

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)

trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

ref_mat = torch.Tensor(np.load(os.path.join('data', 'aux_files', '[J]_function_in_[C].npy')))
remove_part = 1 # hyper parameter is here, the remove limb is pre-computed.
joint_score = ref_mat.sum(1)
part_score = []
for body_part in body_parts:
    # joint_score[:, body_part] = joint_score[:, body_part].mean(dim=-1, keepdim=True)
    part_score.append(joint_score[:, body_part].mean(dim=-1, keepdim=True))
part_score = torch.cat(part_score, dim=-1)
remove_dict = {i: set.union(*[set(body_parts[j]) for j in part_score[i].argsort()[:remove_part]]) for i in range(part_score.shape[0])}

def generate_mask_cpr_feat(x, label, model):
    out = []
    for i in range(x.shape[0]):
        target_joint = remove_dict[label[i].item()]
        pred = model(x[i].unsqueeze(0), target_joint)
        out.append(pred)
    out = torch.cat(out, dim=0)
    return out

# remove_joint = 9 # hyper parameter is here, the remove limb is pre-computed. (the trade-off version)
# max_remove_joint = 20 # only random select remove_joint joints in first max_remove_joint joints.
joint_series = joint_score.argsort(-1)

def generate_mask_cpr_feat_v2(x, label, model, max_remove_joint=20, remove_joint=9):
    cur_keep_joint = joint_series.to(label.device)[label]
    deterministic_keep_part = cur_keep_joint[:, max_remove_joint:]
    rand_idx = torch.randperm(max_remove_joint).to(label.device)[:max_remove_joint - remove_joint]
    target_keep_joint = torch.cat([deterministic_keep_part, cur_keep_joint[:, rand_idx]], dim=1)
    target_keep_joint = target_keep_joint.sort(1)[0]
    out = model(x, target_keep_joint)
    return out

# v3: fuse temporal attention and cpr-based random selection
def generate_mask_cpr_feat_v3(x, label, model, temp_mask=15, max_remove_joint=20, remove_joint=9):
    if temp_mask > 0:
        x = motion_att_temp_mask2(x, temp_mask)
    cur_keep_joint = joint_series.to(label.device)[label]
    deterministic_keep_part = cur_keep_joint[:, max_remove_joint:]
    rand_idx = torch.randperm(max_remove_joint).to(label.device)[:max_remove_joint - remove_joint]
    target_keep_joint = torch.cat([deterministic_keep_part, cur_keep_joint[:, rand_idx]], dim=1)
    target_keep_joint = target_keep_joint.sort(1)[0]
    out = model(x, target_keep_joint)
    return out

def generate_mask_cpr_feat_v2_deterministic(x, label, model, remove_joint=9):
    target_keep_joint = joint_series.to(label.device)[label][:, remove_joint:]
    target_keep_joint = target_keep_joint.sort(1)[0]
    out = model(x, target_keep_joint)
    return out

def generate_mask(model, x):
    # x: n, c, t, v, m
    y, N, M = model.backbone_forward(x) # NM C T V
    mean_feat, _, _ = model.backbone_forward(x.mean(2, keepdims=True).repeat(1, 1, x.shape[2], 1, 1))
    y.requires_grad = True
    with TemporaryGrad():
        z = F.avg_pool2d(y, y.size()[2:])
        z = z.view(N, M, -1).mean(dim=1)
        z = self.fc(z)
        s = (F.normalize(z,dim=-1) * F.normalize(mean_feat,dim=-1)).sum(-1).mean()
        (1-s).backward()
        dz_dy = y.grad
    dz_dy_mean = dz_dy.mean(dim=(2,3))
    cam = dz_dy_mean[:,:,None,None] * y
    cam = F.relu(cam.mean(dim=1, keepdim=True)) / (1e-8)
    for body_part in body_parts:
        cam[:, :, :, body_part] = cam[:, :, :, body_part].mean(dim=-1, keepdim=True)
    cam = cam.clip(0, 1)
    return (cam).float().detach()