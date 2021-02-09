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

class RelationTaskEMB(nn.Module):
    def __init__(self, H, C, out_size):
        super(RelationTaskEMB, self).__init__()
        self.out_size = out_size
        self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
        self.W = torch.nn.Parameter(torch.randn(C * out_size, out_size))
        self.b = torch.nn.Parameter(torch.randn(out_size))
        # self.fc = nn.Linear(C * out_size, out_size)


    def forward(self, memory_vector, class_vector):  # (C,H) (C,H)
        mid_pro = []
        for slice in range(self.out_size):
            slice_inter = torch.mm(torch.mm(memory_vector, self.M[:, :, slice]), class_vector.transpose(1, 0))  # (C,C)
            mid_pro.append(slice_inter)
        mid_pro = torch.cat(mid_pro, dim=0)  # (C*out_size,C) 
        V = torch.mm(F.relu(mid_pro.transpose(0, 1)), self.W) + self.b # (C, C*out_size) @ (C*out_size, out_size) -> (C, out_size)
        return V.T  # (out_size, C)

# class RelationTaskEMB(nn.Module):
#     def __init__(self, H, C, out_size):
#         super(Relation, self).__init__()
#         self.out_size = out_size
#         self.M = torch.nn.Parameter(torch.randn(H, H, out_size))
#         self.W = torch.nn.Parameter(torch.randn(C * out_size, out_size))
#         self.b = torch.nn.Parameter(torch.randn(out_size))

#     def forward(self, class_vector, query_encoder, memory_vector):  # (C,H) (Q,H) (C,H)
#         mid_pro_c = []
#         mid_pro_m = []
#         for slice in range(self.out_size):
#             slice_inter_c = torch.mm(torch.mm(class_vector, self.M[:, :, slice]), query_encoder.transpose(1, 0))  # (C,Q)
#             slice_inter_m = torch.mm(torch.mm(class_vector, self.M[:, :, slice]), memory_vector.transpose(1, 0))  # (C,C)
#             mid_pro_c.append(slice_inter_c)
#             mid_pro_m.append(slice_inter_m)
#         mid_pro_c = torch.cat(mid_pro_c, dim=0)  # (C*out_size,Q)
#         mid_pro_m = torch.cat(mid_pro_m, dim=0)  # (C*out_size,C)
#         V_c = F.relu(mid_pro_c.transpose(0, 1))  # (Q,C*out_size)
#         # V_m = F.relu(mid_pro_m.transpose(0, 1))  # (C,C*out_size)

#         return torch.mm(V_c, mid_pro_m)  # (Q, C)



class DMRInduction(nn.Module):
    def __init__(self, n_class, k_sample, n_query,
                 n_routing, top_k, comp_dim, device):
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
        self.comp_dim = comp_dim
        self.hidden_dim = self.comp_dim
        self.dim_caps = self.hidden_dim // self.n_class
        self.num_caps = n_class
        self.n_routing = n_routing

        self.memory_cls_vector = None

        self.W_comp = nn.Linear(768, self.comp_dim)

        self.DM = DynamicRouting(self.comp_dim, device=self.device)
        self.REL = Relation(H=self.comp_dim, C=n_class, out_size=self.comp_dim) 
        self.REL_TEMB = RelationTaskEMB(H=self.comp_dim, C=n_class, out_size=self.comp_dim) # noshp_fixm
        self.lmda = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, data, attmask, segid, episode=None):
        # (batch[support-5 + query-27], hidden_size=768)
        # pooler_output  = self.encoder(data, attmask, segid)[1]
        hidden  = self.encoder(data, attmask, segid)[2]
        hx = hidden[-2].detach()
        hx = torch.mean(hx, dim=1)
        # checking('pooler_output', pooler_output)
        # [N_class*K_samples, in_dim], [N_class*N_querys, in_dim]
        hx_out = self.W_comp(hx)
        support_encoder, query_encoder = hx_out[0:self.n_support], hx_out[self.n_support:]

        e_c = support_encoder.view(self.n_class, self.k_sample, self.comp_dim) # self.hidden_dim) # modified
        class_vector = self.DM(e_c) # (Q, comp_dim)

        score = self.REL(class_vector, query_encoder) # (Q,C)

        if self.memory_cls_vector is not None:
            memory_n_cur = torch.cat([class_vector, self.memory_cls_vector], dim=1).reshape(self.n_class, 2, -1) # [C, H]
            class_vector_m = self.DM(memory_n_cur)
            # When Only training time update the memory
            taskemb = self.REL_TEMB(self.memory_cls_vector, class_vector) # (out_size, C)
            score_m = torch.mm(query_encoder, taskemb) # (Q, comp_dim) @ (out_size, C) -> (Q, C)
            tot_score = torch.sigmoid(self.lmda*score + (1-self.lmda)*score_m) # add cross term

            if self.training:
                self.memory_cls_vector = class_vector_m.detach()

            
        else:
            self.memory_cls_vector = class_vector.detach() # [C, H]
            tot_score = torch.sigmoid(score)

        return class_vector, tot_score