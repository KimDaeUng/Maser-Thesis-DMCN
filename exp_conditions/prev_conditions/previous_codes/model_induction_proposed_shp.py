import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
from utils import squash, pearsonr, checking
import logging
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
import os
import wandb
import gc

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")
logger = logging.getLogger('debug')
logger.setLevel(logging.WARNING)

class DynamicRouting(nn.Module):
    def __init__(self, hidden_size, device):
        super(DynamicRouting, self).__init__()
        self.hidden_size = hidden_size
        self.l_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.device = device

    def forward(self, encoder_output, iter_routing=3):
        C, K, H = encoder_output.shape
        b = torch.zeros(C, K).to(self.device)
        for _ in range(iter_routing):
            d = F.softmax(b, dim=-1)
            encoder_output_hat = self.l_1(encoder_output)
            c_hat = torch.sum(encoder_output_hat*d.unsqueeze(-1), dim=1)
            c = squash(c_hat)

            b = b + torch.bmm(encoder_output_hat, c.unsqueeze(-1)).squeeze()

        # write.add_embedding(c, metadata=[0, 1, 2, 3, 4],)

        return c


class Relation(nn.Module):
    def __init__(self, H, C, out_size):
        super(Relation, self).__init__()
        self.out_size = out_size
        self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
        self.W = torch.nn.Parameter(torch.randn(C * out_size, C))
        self.b = torch.nn.Parameter(torch.randn(C))

    def forward(self, class_vector, query_encoder):  # (C,H) (Q,H)
        mid_pro = []
        for slice in range(self.out_size):
            slice_inter = torch.mm(torch.mm(class_vector, self.M[:, :, slice]), query_encoder.transpose(1, 0))  # (C,Q)
            mid_pro.append(slice_inter)
        mid_pro = torch.cat(mid_pro, dim=0)  # (C*out_size,Q)
        V = F.relu(mid_pro.transpose(0, 1))  # (Q,C*out_size)
        return torch.mm(V, self.W) + self.b
        # probs = torch.sigmoid(torch.mm(V, self.W) + self.b)  # (Q,C)
        # return probs

# class Relation(nn.Module):
#     def __init__(self, hidden_size, class_num, device, output_size=100, k_max=3):
#         super(Relation, self).__init__()
#         self.class_num = class_num
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.k_max = k_max

#         self.M = nn.init.xavier_normal_(torch.FloatTensor(hidden_size, hidden_size, self.output_size))
#         self.M.requires_grad = True

#         # self.l1 = nn.Linear(self.hidden_size, 1)
#         self.l2 = nn.Linear(self.output_size, 1)

#         self.fc = nn.Linear(self.hidden_size*self.output_size, class_num, bias=True)

#         self.device = device


    # def forward(self, class_vector, query_vector):
    #     print("cls, query shape")
    #     print(class_vector.shape)
    #     print(query_vector.shape)
    #     query_size, hidden_size = query_vector.shape
    #     # query_vector = query_vector.unsqueeze(1).repeat(1, self.class_num, 1)
    #     # class_vector = class_vector.unsqueeze(0).repeat(query_size, 1, 1)
    #     # print("cls, query shape after")
    #     # print(class_vector.shape)
    #     # print(query_vector.shape)
    #     # concatenate
    #     # concatenated = torch.cat([query_vector, class_vector], dim=-1)

    #     # concatenated = self.l1(concatenated).squeeze()
    #     # concatenated = concatenated.view(self.output_size, query_size, self.class_num)
    #     mid_pro = []
    #     for i in range(self.output_size):
    #         v = self.M[:, :, i]
    #         inter = torch.mm(query_vector, torch.mm(class_vector, v).transpose(0, 1))

    #         # inter = torch.sum(torch.mul(torch.matmul(class_vector.unsqueeze(-1), query_vector.unsqueeze(2)), v), dim=2)
    #         # inter= torch.sum(inter*v, dim=1)
    #         # inter = (self.M[:, :, i].unsqueeze(0).repeat(query_size, 1, 1))*torch.bmm(query_vector, class_vector.transpose(1, 2))
    #         mid_pro.append(inter)
    #     tensor_bi_product = torch.stack(mid_pro, dim=0)  # (C*K,Q)
    #     print("tensor_bi_product")
    #     print(tensor_bi_product.shape)
    #     activate = F.relu(tensor_bi_product)
    #     reshape = activate.permute(1, 2, 0)
    #     other = self.l2(reshape).squeeze()
    #     print("other")
    #     print(other)
    #     # other = self.l1(other).view(query_size, self.class_num)
    #     # tensor_bi_product = concatenated+tensor_bi_product

    #     # k_max = k_max_pooling(F.tanh(tensor_bi_product), dim=2, k=self.k_max).view(query_size, self.k_max*self.output_size)
    #     # output = self.fc(F.relu(tensor_bi_product))
    #     probability = torch.sigmoid(other)
    #     return probability
        

class DMRInduction(nn.Module):
    def __init__(self, n_class, k_sample, n_query,
                 memory, n_routing, top_k, comp_dim, device):
        super(DMRInduction, self).__init__()

        # Params of N-way K-shot setting
        self.n_class = n_class
        self.k_sample = k_sample
        self.n_support = self.n_class * self.k_sample
        self.n_query = n_query
        self.device = device

        # Parames of Encoder 
        self.encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True) # path?
        # To check the bert input, See the Sandbox3.ipynb in google drive
        
        # Freeze the BERT
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Parames of Capsule Networks
        self.memory = memory.detach()
        self.n_memory = memory.shape[0] # (N_memory, hidden_dim=768) 
        self.comp_dim = comp_dim
        self.hidden_dim = self.comp_dim
        self.dim_caps = self.hidden_dim // self.n_class
        self.num_caps = n_class
        self.n_routing = n_routing

        self.memory_cls_vector = None

        self.W_comp = nn.Linear(768, self.comp_dim)

        self.DM = DynamicRouting(self.comp_dim, device=self.device)
        # self.REL = Relation(hidden_size=self.comp_dim, class_num=n_class, device=self.device, output_size=self.comp_dim) 
        self.REL = Relation(H=self.comp_dim, C=n_class, out_size=self.comp_dim) 
        # self.REL_TASK = Relation(H=self.comp_dim, C=n_class, out_size=self.comp_dim) # noshp_fixm


    def forward(self, data, attmask, segid, episode=None):
        # (batch[support-5 + query-27], hidden_size=768)
        # pooler_output  = self.encoder(data, attmask, segid)[1]
        hidden  = self.encoder(data, attmask, segid)[2]
        hx = hidden[-2].detach()
        hx = torch.mean(hx, dim=1)
        # checking('pooler_output', pooler_output)
        # [N_class*K_samples, in_dim], [N_class*N_querys, in_dim]
        support_encoder_, query_encoder_ = hx[0:self.n_support], hx[self.n_support:]
        support_encoder = self.W_comp(support_encoder_)
        query_encoder = self.W_comp(query_encoder_)

        e_c = support_encoder.view(self.n_class, self.k_sample, self.comp_dim) # self.hidden_dim) # modified
        class_vector = self.DM(e_c)

        score = self.REL(class_vector, query_encoder)

        if self.memory_cls_vector is not None:
            memory_n_cur = torch.cat([class_vector, self.memory_cls_vector], dim=1).reshape(self.n_class, 2, -1) # [C, H]
            class_vector_m = self.DM(memory_n_cur)
            # When Only training time update the memory
            if self.training:
                self.memory_cls_vector = class_vector_m.detach()
            score_m = self.REL(class_vector_m, query_encoder)    # shp_fixm
            # score_m = self.REL_TASK(class_vector_m, query_encoder) # noshp_fixm
            tot_score = torch.sigmoid(score + score_m)
        else:
            self.memory_cls_vector = class_vector.detach() # [C, H]
            tot_score = torch.sigmoid(score)

        return class_vector, tot_score