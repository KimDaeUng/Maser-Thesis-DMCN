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


# class Relation(nn.Module):
#     def __init__(self, H, C, out_size):
#         super(Relation, self).__init__()
#         self.out_size = out_size
#         self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
#         self.W = torch.nn.Parameter(torch.randn(C * out_size, C))
#         self.b = torch.nn.Parameter(torch.randn(C))

#     def forward(self, class_vector, query_encoder):  # (C,H) (Q,H)
#         mid_pro = []
#         for slice in range(self.out_size):
#             slice_inter = torch.mm(torch.mm(class_vector, self.M[:, :, slice]), query_encoder.transpose(1, 0))  # (C,Q)
#             mid_pro.append(slice_inter)
#         mid_pro = torch.cat(mid_pro, dim=0)  # (C*out_size,Q)
#         V = F.relu(mid_pro.transpose(0, 1))  # (Q,C*out_size)
#         return torch.mm(V, self.W) + self.b

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size=32, hidden_size=16):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1) # n_class

    def forward(self, a, b):  # 10*100  #valid 25*100

        n_examples = a.size()[0] * a.size()[1]
        a = a.reshape(n_examples, -1)
        b = b.reshape(n_examples, -1)
        # cosine = F.cosine_similarity(a, b)
        # print(cosine.shape)
        # return cosine

        # not converage
        x = torch.cat((a, b), 1)
        x = F.relu(self.fc1(x))  # hiddensize->hiddensize
        x = self.fc2(x)  # hiddensize->scale
        return x
        

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
        self.REL = RelationNetwork(input_size=2*self.comp_dim) 
        self.REL_TASK = RelationNetwork(input_size=2*self.comp_dim) 


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

        class_vector_ext = class_vector.unsqueeze(0).repeat(query_encoder.shape[0], 1, 1)
        query_encoder_ext = query_encoder.unsqueeze(0).repeat(self.n_class, 1, 1).permute(1, 0, 2)

        score = self.REL(class_vector_ext, query_encoder_ext).view(-1, self.n_class)
        if self.memory_cls_vector is not None:
            memory_n_cur = torch.cat([class_vector, self.memory_cls_vector], dim=1).reshape(self.n_class, 2, -1) # [C, H]
            class_vector_m = self.DM(memory_n_cur)
            class_vector_m_ext = class_vector_m.unsqueeze(0).repeat(query_encoder.shape[0], 1, 1)
            if self.training:
                self.memory_cls_vector = class_vector_m.detach()
            score_m = self.REL_TASK(class_vector_m_ext, query_encoder_ext).view(-1, self.n_class)
            
            tot_score = torch.sigmoid(score + score_m)

        else:
            self.memory_cls_vector = class_vector.detach() # [C, H]
            tot_score = torch.sigmoid(score)

        return class_vector, tot_score