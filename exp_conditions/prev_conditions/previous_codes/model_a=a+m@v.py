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

class DynamicMemoryInduction(nn.Module):
    def __init__(self, in_caps, in_dim, 
                       num_caps, dim_caps,
                       num_routing,
                       W_share,
                       n_class, k_sample, device:torch.device):
        super(DynamicMemoryInduction, self).__init__()
        self.in_dim = in_dim  # input capsule dimensionality (i.e. length)
        self.in_caps = in_caps
        self.dim_caps = dim_caps
        self.num_caps = num_caps # the number of high-level capsules == n_class 
        self.num_routing = num_routing
        self.beta = float(config['Model']['temperature'])

        # N = n_class, K = k_samples 
        # M = in_caps # the number of memories
        self.n_class = n_class
        self.k_sample = k_sample # QIM for n_querys
        self.n_support = self.n_class*self.k_sample

        self.device = device
        self.W = nn.Linear(self.in_dim, self.dim_caps*self.num_caps)


    def forward(self, m, q):
        '''
        m : W_base, memory [N_of_memory(or N_way*K_shot(or N_query)), d_dim=768]
        q : sample vector to be adapted to W_base [N_class*K_samples, d_dim=768]
        '''        
        # W @ x =
        # # Batch = N_memory or N_clss * K_sample(N_query)
        # (Batch, num_caps*dim_caps, in_dim) @ (Batch, in_dim) =
        # (Batch, num_caps*dim_caps)

        # hat_m_ij = squash(self.W(m).reshape(self.in_caps, self.num_caps, self.dim_caps))
        # hat_q_ij = squash(self.W(q).reshape(q.shape[0], self.num_caps, self.dim_caps))
        # self.in_caps = m.shape[0]
        hat_m_ij = self.W(m).reshape(self.in_caps, self.num_caps, self.dim_caps)
        hat_q_ij = self.W(q).reshape(q.shape[0], self.num_caps, self.dim_caps)
        # checking('m', m)
        # checking('self.W', self.W.weight)
        # checking('hat_m_ij', hat_m_ij)
        # checking('hat_q_ij', hat_q_ij)
        
        # 배치 차원으로 복사(q.shape[0])
        tmp_hat_m_ij = hat_m_ij.unsqueeze(0).expand(q.shape[0], self.in_caps, self.num_caps, self.dim_caps).detach()
        # tmp_hat_q_ij = hat_q_ij.unsqueeze(1).repeat(1, self.in_caps, 1, 1).detach()
        # Momory 차원으로 복사(self.in_caps)
        tmp_hat_q_ij = hat_q_ij.unsqueeze(1).expand(q.shape[0], self.in_caps, self.num_caps, self.dim_caps).detach()


        # checking('tmp_hat_m_ij', tmp_hat_m_ij)
        # checking('tmp_hat_q_ij', tmp_hat_q_ij)
        
        a = torch.zeros(q.shape[0], self.in_caps, self.num_caps).to(self.device)
        # checking('a', a)

        # pearson corr
        p_ij = torch.tanh(pearsonr(tmp_hat_m_ij, tmp_hat_q_ij).detach()).squeeze(-1)
        # checking('p_ij', p_ij)

        # Dynamic Memory Routing Algorithm
        for routing_iter in range(self.num_routing-1):
            d = F.softmax(a.detach()/self.beta, dim=2) 
            # checking('d', d, True)

            d_sum_p = torch.add(d, p_ij)
            # checking('d_sum_p', d_sum_p, True)

            hat_v = torch.mul(tmp_hat_m_ij, d_sum_p.unsqueeze(-1)).sum(dim=1)

            v = squash(hat_v)
            # checking('v', v, True)

            # dot product agreement between the v_j and the m_ij
            hat_m_cos_v = torch.sum(torch.mul(
                            tmp_hat_m_ij,
                            v.unsqueeze(1).expand(
                                q.shape[0], self.in_caps, self.num_caps, self.dim_caps
                                )), dim=-1)

            # checking('hat_m_cos_v', hat_m_cos_v, True)

            a = a + torch.mul(p_ij, hat_m_cos_v)
            # checking('a - routing loop', a, True)

            tmp_hat_q_ij = torch.div(
                torch.add(
                    tmp_hat_q_ij,
                     v.unsqueeze(1).expand(
                         q.shape[0], self.in_caps, self.num_caps, self.dim_caps
                         )
                         ), 2)
            # checking('tmp_hat_q_ij', tmp_hat_q_ij, True)

            p_ij = torch.tanh(pearsonr(tmp_hat_m_ij, tmp_hat_q_ij).detach()).squeeze(-1)
            # checking('p_ij', p_ij, True)


        # after routing ....
        d = F.softmax(a.detach()/self.beta, dim=2) # .to(self.device)
        # checking('d', d)

        d_sum_p = d + p_ij
        # checking('d_sum_p', d_sum_p)

        hat_v = torch.sum(
                    torch.mul(
                        hat_m_ij.unsqueeze(0).expand(
                            q.shape[0], self.in_caps, self.num_caps, self.dim_caps),
                        d_sum_p.detach().unsqueeze(-1)
                        ), dim=1)
        # checking('hat_v', hat_v)

        q_prime = squash(hat_v).reshape(q.shape[0], -1) # gradinet?
        # checking('q_prime', q_prime)

        return q_prime

class DynamicTaskMemoryInduction(nn.Module):
    def __init__(self, in_caps, in_dim, 
                       num_caps, dim_caps,
                       num_routing,
                       W_share,
                       n_class, k_sample, device:torch.device):
        super(DynamicTaskMemoryInduction, self).__init__()
        self.in_dim = in_dim  # input capsule dimensionality (i.e. length)
        self.in_caps = in_caps
        self.dim_caps = dim_caps
        self.num_caps = num_caps # the number of high-level capsules == n_class 
        self.num_routing = num_routing
        self.beta = float(config['Model']['temperature'])

        # M = in_caps # the number of memories
        self.n_class = n_class
        self.k_sample = k_sample # QIM for n_querys
        self.n_support = self.n_class*self.k_sample

        self.device = device

        self.W = nn.Linear(self.in_dim, self.num_caps*self.dim_caps) # original
        # self.W = nn.Parameter(0.01*torch.randn(self.in_dim, self.num_caps*self.dim_caps)) # add
        # self.register_parameter("DTMR_Wj", self.W)

        self.log_interval = int(config['Log']['log_interval'])
        
    def forward(self, m, q, episode=None):
        '''
        m : W_base, memory [N_of_memory(or N_way*K_shot(or N_query)), d_dim=768]
        q : sample vector to be adapted to W_base [N_class*K_samples, d_dim=768]
        입력 값 변경(M : Support set or Query set // q : Most Relevant Memory)
        '''        
        # W @ x =
        # # Batch = N_memory or N_clss * K_sample(N_query)
        # (Batch, num_caps*dim_caps, in_dim) @ (Batch, in_dim) =
        # (Batch, num_caps*dim_caps)

        # self.in_caps = m.shape[0]
        hat_m_ij = self.W(m).reshape(self.in_caps, self.num_caps, self.dim_caps) 
        hat_q_ij = self.W(q).reshape(q.shape[0], self.num_caps, self.dim_caps)

        # checking('m', m)
        # checking('self.W', self.W.weight)
        # checking('hat_m_ij', hat_m_ij)
        # checking('hat_q_ij', hat_q_ij)
        
        # Expand가 불필요할 수 있음, q는 벡터 하나만 받기 때문
        # q를 하나 받아도 코드가 달라지지 않기 때문에 그대로 둠 
        # 배치 차원으로 복사(q.shape[0])
        tmp_hat_m_ij = hat_m_ij.unsqueeze(0).expand(q.shape[0], self.in_caps, self.num_caps, self.dim_caps).detach()
        # tmp_hat_q_ij = hat_q_ij.unsqueeze(1).repeat(1, self.in_caps, 1, 1).detach()
        # Momory 차원으로 복사(self.in_caps)
        tmp_hat_q_ij = hat_q_ij.unsqueeze(1).expand(q.shape[0], self.in_caps, self.num_caps, self.dim_caps).detach()
        # checking('tmp_hat_m_ij', tmp_hat_m_ij)
        # checking('tmp_hat_q_ij', tmp_hat_q_ij)
        
        a = torch.zeros(q.shape[0], self.in_caps, self.num_caps).to(self.device)
        # checking('a', a)

        # pearson corr
        p_ij = torch.tanh(pearsonr(tmp_hat_m_ij, tmp_hat_q_ij).detach()).squeeze(-1)
        # checking('p_ij', p_ij)
        
        if (config['Log']['log_DMTR'] == 'y')&(episode % self.log_interval == 0):
            file_ = {"p_ij" : p_ij.detach().cpu().numpy() }
            file = { 0 : file_ }

        # Dynamic Memory Routing Algorithm
        for routing_iter in range(self.num_routing-1):
            # print("\tp_ij : ------------------------\n",-p_ij)
            # a : [N_class*N_query, N_memory(N of Low-level capsule), N_class(N of High-level capsule)]
            d = F.softmax(a.detach()/self.beta, dim=2) # .to(self.device)
            # checking('d', d, True)
            d_sum_p = torch.add(d, -p_ij)
            # print("\td_sum_p : ------------------------\n",-d_sum_p)

            # checking('d_sum_p', d_sum_p, True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

            hat_v = torch.mul(tmp_hat_m_ij, d_sum_p.unsqueeze(-1)).sum(dim=1) 

            v = squash(hat_v)
            # checking('v', v, True)

            # dot product agreement between the v_j and the m_ij
            hat_m_cos_v = torch.sum(torch.mul(
                            tmp_hat_m_ij,
                            v.unsqueeze(1).expand(
                                q.shape[0], self.in_caps, self.num_caps, self.dim_caps)
                                ), dim=-1)
            # print("\that_m_cos_v : ------------------------\n",-hat_m_cos_v)

            # checking('hat_m_cos_v', hat_m_cos_v, True)

            # hat_q_cos_v
            # a = a + torch.mul(-p_ij, hat_m_cos_v)
            a = a + hat_m_cos_v
            # print("\t @@a : ------------------------\n",-a)

            # a = a + hat_m_cos_v
            # checking('a - routing loop', a, True)
            
            # 이게 학습을 방해하나?
            tmp_hat_q_ij = torch.div(
                torch.add(
                    tmp_hat_q_ij,
                     v.unsqueeze(1).expand(
                        q.shape[0], self.in_caps, self.num_caps, self.dim_caps
                        )
                        ), 2)
            # checking('tmp_hat_q_ij', tmp_hat_q_ij, True)

            p_ij = torch.tanh(pearsonr(tmp_hat_m_ij, tmp_hat_q_ij).detach()).squeeze(-1)
            # checking('p_ij', p_ij, True)

            if (config['Log']['log_DMTR'] == 'y')&(episode % self.log_interval == 0):
                file[routing_iter+1] = {}
                file[routing_iter+1]['p_ij'] = p_ij.detach().cpu().numpy()
                file[routing_iter+1]['a'] = a.detach().cpu().numpy()
                file[routing_iter+1]['d'] = d.detach().cpu().numpy()
                file[routing_iter+1]['d_sum_p'] = d_sum_p.detach().cpu().numpy()
                file[routing_iter+1]['hat_v'] = hat_v.detach().cpu().numpy()
                file[routing_iter+1]['v'] = v.detach().cpu().numpy()
                file[routing_iter+1]['m@v'] = hat_m_cos_v.detach().cpu().numpy()
                file[routing_iter+1]['tmp_hat_q_ij'] = tmp_hat_q_ij.detach().cpu().numpy()

        # after routing ....
        # print("="*50)
        # print("P_ij : ------------------------\n",p_ij)
        d = F.softmax(a.detach() /self.beta, dim=2) # .to(self.device)
        # checking('d', d)

        d_sum_p = torch.add(d, -p_ij)
        # print("d_sum_p : ------------------------\n",d_sum_p)

        # checking('d_sum_p', d_sum_p)

        hat_v = torch.sum(
                    torch.mul(
                        hat_m_ij.unsqueeze(0).expand(
                            q.shape[0], self.in_caps, self.num_caps, self.dim_caps),
                        d_sum_p.unsqueeze(-1)
                        ), dim=1) # original
        
        # checking('hat_v', hat_v)

        q_prime = squash(hat_v).reshape(q.shape[0], -1) # gradinet?
        # checking('q_prime', q_prime)

        if (config['Log']['log_DMTR'] == 'y')&(episode % self.log_interval == 0):
            file[routing_iter+2] = {}
            file[routing_iter+2]['d'] = d.detach().cpu().numpy()
            file[routing_iter+2]['d_sum_p'] = d_sum_p.detach().cpu().numpy()
            file[routing_iter+2]['hat_v'] = hat_v.detach().cpu().numpy()
            file[routing_iter+2]['q_prime'] = q_prime.detach().cpu().numpy()
            torch.save(file, os.path.join(config['Log']['log_value_path'], 'DMTR_{}.dict'.format(episode)))
            del file, file_
            gc.collect()


        return q_prime


class SimilarityClassifier(nn.Module):
    def __init__(self, n_class, n_query, hidden_dim, device):
        super(SimilarityClassifier, self).__init__()
        self.n_class = n_class
        self.n_query = n_query 
        self.n_samples = self.n_class * self.n_query
        self.hidden_dim = hidden_dim
        self.device = device
        
        # self.tau = torch.ones(1)
        # self.tau = torch.nn.Parameter(torch.ones(1))
        # self.register_parameter("Sim_tau", self.tau)
        # self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, e_c, e_q): 
        # (N_class*(N_class*N_query),H) (N_class*N_query,H) ?
        # [TODO] N_query는 입력에 따라 유동적이게 바꿔야함
        # 각각의 Query에 대해서 2개씩 Claass vector를 받게됨

        # e_c_norm = F.normalize(e_c, p=2, dim=-1).view(self.n_class, e_q.shape[0], self.hidden_dim)
        # checking('e_c in', e_c)
        # e_c_norm.requires_grad = True

        # [n_class*n_class*n_query, in_dim]
        e_c_norm = torch.norm(e_c, p=2, dim=-1).detach()
        e_c_after = e_c.div(e_c_norm.detach().unsqueeze(-1).expand_as(e_c))
        # checking('e_c_after', e_c_after)
        # print('e_c')
        # print(e_c_after.shape)
        # print(e_c_after)

        # (N_class*N_query, H)
        e_q_norm = torch.norm(e_q, p=2, dim=-1).detach()
        e_q_after = e_q.div(e_q_norm.detach().unsqueeze(-1).expand_as(e_q))
        # e_q_norm = F.normalize(e_q, p=2, dim=-1)
        # checking('e_q_after', e_q_after)
        # print('e_q')
        # print(e_q_after.shape)
        # print(e_q_after)

        # logger.info("e_c : {}".format(e_c.shape))
        # logger.info("e_q : {}".format(e_c.shape))
        
        # (N_class, N_class * N_query, H),  (N_class * N_query, H)
        # s_qc_b = F.cosine_similarity(e_c, e_q, dim=-1)
        s_qc_b = torch.mul(
            e_c_after.reshape(
                self.n_class, e_q.shape[0], -1
                )
                , e_q_after).sum(dim=-1).T
        # checking('s_qc_b', s_qc_b)
        # print('s_qc_b')
        # print(s_qc_b.shape)
        # print(s_qc_b)

        # (N_class, N_class * N_query)        
        # logger.info("s_qc_b : {}".format(s_qc_b.shape))
        # s_qc = torch.mul(self.tau, s_qc_b)
        
        # checking('s_qc', s_qc)
        # print('s_qc')
        # print(s_qc.shape)
        # print(s_qc)
        # (N_class * N_query, N_class)        
        # probs = F.softmax(s_qc.T,dim=1)
        # s_qc = s_qc.T
        return s_qc_b

class TaskRetrival(nn.Module):
    def __init__(self, memory, top_k):
        super(TaskRetrival, self).__init__()
        self.memory = memory
        self.top_k = top_k
        # self.n_memory = self.top_k * 2
        self.n_memory = self.top_k

    # def forward(self, x):
    #     '''
    #     x : encodered input : [N_class*K_samples, H_dim]
    #     memory : [N_memory, H_dim]
    #     '''
    #     task_emb = torch.mean(x.detach(), dim=0, keepdim=True) # [H_dim]
    #     cossim_table = F.cosine_similarity(self.memory, task_emb)

    #     pos_idx = torch.topk(cossim_table, self.top_k).indices
    #     neg_idx = torch.topk(-cossim_table, self.top_k).indices
    #     retrieved_m = self.memory[[pos_idx, neg_idx], : ]
    #     retrieved_m[1] = -retrieved_m[1]
    #     return retrieved_m
    def forward(self, x):
        '''
        x : encodered input : [N_class*K_samples, H_dim]
        memory : [N_memory, H_dim]
        '''
        task_emb = torch.mean(x.detach(), dim=0, keepdim=True) # [H_dim]
        cossim_table = F.cosine_similarity(self.memory, task_emb)

        pos_idx = torch.topk(cossim_table, self.top_k).indices
        # neg_idx = torch.topk(-cossim_table, self.top_k).indices
        retrieved_m = self.memory[pos_idx, : ]
        return retrieved_m
        


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
        self.encoder
        
        # Parames of Capsule Networks
        self.memory = memory.detach()
        self.n_memory = memory.shape[0] # (N_memory, hidden_dim=768) 
        self.comp_dim = comp_dim
        self.hidden_dim = self.comp_dim
        self.dim_caps = self.hidden_dim // self.n_class
        self.num_caps = n_class
        self.n_routing = n_routing
        self.top_k = top_k

        self.emb_save_interval = int(config['Log']['emb_save_interval'])

        # self.W_share = nn.Parameter(0.01 * torch.randn(1, self.n_class, 1, self.dim_caps, self.hidden_dim)).to(self.device)
        # self.W_share2 = nn.Parameter(0.01 * torch.randn(1, self.n_class, 1, self.dim_caps, self.hidden_dim)).to(self.device)
        # self.W_share2 = nn.Parameter(0.01 * torch.randn(1, self.n_class, 1, self.dim_caps, self.hidden_dim)).to(self.device)

        self.TaskRetrieval = TaskRetrival(self.memory, top_k=self.top_k)
        
        # W_comp -> 다음 시도용 준비 768 차원 -> 128차원 축소하여 처리 
        self.W_comp = nn.Linear(768, self.comp_dim)
        # self.W_share = nn.Linear(self.hidden_dim, self.dim_caps*self.num_caps)
    
        self.DTMR = DynamicTaskMemoryInduction(
             in_caps=self.TaskRetrieval.n_memory, in_dim=self.comp_dim, # self.hidden_dim, # in_caps 는 m.shape[0]으로 처리 (쿼리, 서포트 다 M에 받기 때문에 입력값 가변적)
             num_caps=self.n_class, dim_caps=self.dim_caps,
             num_routing=self.n_routing, W_share=None,
             n_class=self.n_class, k_sample=self.k_sample,
             device=self.device)
        
        # It differ with DMR at the number of the input capsules
        self.QIM = DynamicMemoryInduction(
             in_caps=self.k_sample, in_dim=self.comp_dim, # self.hidden_dim,
             num_caps=self.n_class, dim_caps=self.dim_caps,
             num_routing=self.n_routing, W_share=None,
             n_class=self.n_class, k_sample=self.n_query,
             device=self.device)

        self.simclassifier = SimilarityClassifier(self.n_class, self.n_query, self.comp_dim, self.device)#self.hidden_dim, self.device)

    def forward(self, data, attmask, segid, episode=None):
        # (batch[support-5 + query-27], hidden_size=768)
        # pooler_output  = self.encoder(data, attmask, segid)[1]
        hidden  = self.encoder(data, attmask, segid)[2]
        hx = hidden[-2].detach()
        hx = torch.sum(hx, dim=1)
        # checking('pooler_output', pooler_output)
        # [N_class*K_samples, in_dim], [N_class*N_querys, in_dim]
        support_encoder_, query_encoder_ = hx[0:self.n_support], hx[self.n_support:]
        support_encoder = self.W_comp(support_encoder_)
        query_encoder = self.W_comp(query_encoder_)


        # Dynamic Memory Routing Process
        # Input :
        #   - m : W_base, memory [N_of_memory, d_dim=768]
        #   - q : sample vector to be adapted to W_base [N_class*K_samples, d_dim=768]
        # Output :
        #   - e_cs : [N_class * K_samples, in_dim]
        # support_encoder.requires_grad = True
        # query_encoder.requires_grad = True

        ret_memory_ = self.TaskRetrieval(support_encoder_)
        ret_memory = self.W_comp(ret_memory_)


        e_cs = self.DTMR(ret_memory.detach(), support_encoder, episode).view(self.n_class, self.k_sample, self.comp_dim) # self.hidden_dim) # modified
        e_cq = self.DTMR(ret_memory.detach(), query_encoder, episode) # add  # [N_class*N_query]
        
        # checking("e_cs_", e_cs_)
        # checking("e_cs", e_cs)

        # Query-enhanced Induction Module
        # Batch Size 쪽에서 불일치 발생가능성
        # Input :
        #   - e_cs : adapted sample vectors
        #   - query_encoder : sample vector to be adapted to q_prime [N_class*N_querys, d_dim=768]
        # Output :
        #   - e_c# : [N_class*N_querys, in_dim]
        # -> query vector 마다 N_class 개수 만큼의 e_c를 얻음
        
        e_c0 = self.QIM(e_cs[0, :, :], e_cq)
        e_c1 = self.QIM(e_cs[1, :, :], e_cq)

        # print('after QIM-----------')
        # checking("e_c0", e_c0)
        # checking("e_c1", e_c1)


        # [N_class, N_class*N_querys, in_dim]
        e_c_end = torch.cat([e_c0, e_c1], dim=0)

        logits = self.simclassifier(e_c_end, e_cq)
        # checking("logits", logits)

        if (config['Log']['log_DMTR']=='y')&self.training&(episode % self.emb_save_interval == 0):           
            file = {'support_encoder' : support_encoder.detach().cpu().numpy(),
                    'query_encoder' : query_encoder.detach().cpu().numpy(),
                    'ret_memory' : ret_memory.detach().cpu().numpy(),
                    'e_cs' : e_cs.detach().cpu().numpy(),
                    'e_cq' : e_cq.detach().cpu().numpy(),
                    'e_c_end' : e_c_end.detach().cpu().numpy(),
                    'logits' : logits.detach().cpu().numpy(),
                    }
            torch.save(file, os.path.join(config['Log']['emb_path'], "_{}.pt".format(episode)))
            del file
            gc.collect()
        
        return logits