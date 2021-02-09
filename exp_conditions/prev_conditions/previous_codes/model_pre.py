import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel
from utils import squash, pearsonr
import logging
import numpy as np

logger = logging.getLogger('debug')
logger.setLevel(logging.WARNING)

# def checking(name, m, indent=False):
#     pass



def checking(name, m, indent=False):
    if indent:
        print("\t{} - shape {}".format(name, m.shape))
        print("\t{} - is nan {}".format(name, torch.isnan(m).sum()))
        print("\t{} - is inf {}".format(name, torch.isinf(m).sum()))
        print("\t{} - max : {} / min : {} ".format(name, torch.max(m),torch.min(m)))
    else:
        print("{} - shape {}".format(name, m.shape))
        print("{} - is nan {}".format(name, torch.isnan(m).sum()))
        print("{} - is inf {}".format(name, torch.isinf(m).sum()))
        print("{} - max : {} / min : {} ".format(name, torch.max(m),torch.min(m)))


class DynamicMemoryInduction(nn.Module):
    def __init__(self, in_caps, in_dim, 
                       num_caps, dim_caps,
                       num_routing,
                       W_share,
                       n_class, k_samples, device:torch.device):
        super(DynamicMemoryInduction, self).__init__()
        self.in_dim = in_dim  # input capsule dimensionality (i.e. length)
        self.in_caps = in_caps
        self.dim_caps = dim_caps
        self.num_caps = num_caps # the number of high-level capsules == n_class 
        self.num_routing = num_routing

        # N = n_class, K = k_samples 
        # M = in_caps # the number of memories
        self.n_class = n_class
        self.k_samples = k_samples # QIM for n_querys
        self.n_support = self.n_class*self.k_samples
        # self.origin_memory == origin_memory
        # if self.origin_memory == True:
        #     self.n_memory = 

        # self.n_memory = 

        self.device = device

        # in_caps가 memory를 받을 때와 Query를 받을 때가 달라지므로 (X) 같음, memory는 support set의 개수
        # 가중치 공유를 위해 1로 두고 in_caps를 각각 다르게 주어 별도의 오브젝트로 만듦
        # self.W_share = nn.Parameter(0.01 * torch.randn(1, self.num_caps, 1, self.dim_caps, self.in_dim))
        # self.W_share = 0.01 * torch.randn(1, self.num_caps, 1, self.dim_caps, self.in_dim)
        self.W_share = 0.01 * torch.randn(1, self.num_caps, 1, self.dim_caps, self.in_dim)
        self.W = nn.Parameter(self.W_share.expand(1, self.num_caps, self.in_caps, self.dim_caps, self.in_dim).contiguous())
        # Add for bias
        self.b = nn.Parameter(torch.ones(1, self.num_caps, self.in_caps, self.dim_caps))
        self.register_parameter("DMR_Wj", self.W)
        self.register_parameter("DMR_Wj_bias", self.b)
        # self.tanh = torch.nn.Tanh

        # @@@ New 
        self.W_n = nn.Parameter(torch.randn(self.in_dim, self.dim_caps)) # 768 -> 384
        self.W_b_n = nn.Parameter(torch.randn(self.in_dim, 1)) 


    def forward(self, m, q):
        '''
        m : W_base, memory [N_of_memory, d_dim=768]
        q : sample vector to be adapted to W_base [N_class*K_samples, d_dim=768]
        '''
        # (N*K (or M), H)
        # m = m.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(q.shape[0], self.num_caps, self.in_caps, self.dim_caps, 1).detach().clone()
        # m = m.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(q.shape[0], 1, self.in_caps, self.in_dim, 1) # / self.in_dim
        
        
        m = torch.mm(m, self.W_n).reshape(self.in_caps, self.dim_caps)
        # checking('m', m)
        # m = m.unsqueeze(0).unsqueeze(0).expand(m.shape[0], self.in_caps, self.in_dim).detach()
        
        # Initialize
        # W @ x =
        # (1, self.num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, self.num_caps, in_caps, dim_caps, 1)
        # logger.info("m.shape : {}".format(m.shape))
        # logger.info("W.shapt : {}".format(self.W.shape))
        m.requires_grad = True
        print("self.W - is_leaf" ,  self.W.is_leaf)

        hat_m_ij_ = torch.matmul(self.W, m).squeeze(-1)
        hat_m_ij_.requires_grad = True
        hat_m_ij = (hat_m_ij_, self.b)
        # hat_m_ij.requires_grad = True
        print(hat_m_ij.is_contiguous())
        print("!!hat_m_ij : ", hat_m_ij.requires_grad)
        print("hat_m_ij - is_leaf",hat_m_ij.is_leaf)
        # checking('W', self.W)


        # logger.info(hat_m_ij.shape)

        # hat_m_ij = hat_m_ij.squeeze(-1)
        # checking('hat_m_ij', hat_m_ij)
        # logger.info(hat_m_ij.shape)
        # detach hat_m_ij during routing iteration to prevent gradient from flowing
        tmp_hat_m_ij = hat_m_ij.detach()
        print("tmp_hat_m_ij - is_leaf",tmp_hat_m_ij.is_leaf)

        # logger.info("hat_m_ij : {}".format(tmp_hat_m_ij.shape))

        # Sample vector q s 
        # (n_class, k_samples, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        print("q before reshape - is_leaf",q.is_leaf)
        q_ = q.reshape(-1, 1, self.in_dim, 1)
        # q = q.reshape(q.shape[0], 1, self.in_dim, 1)
        print("q : ", q_.is_contiguous())
        print("q - is_leaf",q_.is_leaf)

        # logger.info(q.shape)
        # each sample vectors are copied to each memory but it should have diffrent memory address
        # when check with id function, it shows same id but
        # when use repeat, if you change the one of copies, they are dependent to each other
        # Above code, for memory, m is imutable value so, we don't need to use repeat to that. 
        # -> 하지만 다시 expand를 사용한다. 어차피 q_ij에서 i 즉 input_capsule(memory 개수)는
        #    계산을 위해 값이 차원따라서 반복만 될 뿐 같은 값이 되어야함
        # q = q.unsqueeze(2).repeat(1, 1, in_caps, 1, 1)
        # 수정함 : 들어오는 batch size에 맞게 expand 시켜야하므로.
        q__ = q_.unsqueeze(2).expand(q.shape[0], 1, self.in_caps, self.in_dim, 1) # Dont use this
        # logger.info(q.shape)
        # checking('q', m)
        q__.requires_grad = True
        print("q : ", q__.is_contiguous())
        print("q - after expand - is_leaf",q.is_leaf)
        # W @ x =
        # (1, self.num_caps, self.in_caps, dim_caps, in_dim) @ (batch_size, 1, self.in_caps, in_dim, 1) =
        # (batch_size, 1, self.num_caps, self.in_caps, dim_caps, 1)
        hat_q_ij = torch.matmul(self.W, q__)
        print("hat_q_ij : ", hat_q_ij.is_contiguous())
        # logger.info(hat_q_ij.shape)
        print("hat_q_ij : ", hat_q_ij.is_leaf)
        print("W - is_leaf",self.W.is_leaf)

        hat_q_ij = hat_q_ij.squeeze(-1)
        # logger.info(hat_q_ij.shape)
        # checking('hat_q_ij', hat_q_ij)
        print("hat_q_ij : ", hat_q_ij.is_contiguous())
        print("hat_q_ij squee - is_leaf",hat_q_ij.is_leaf)

        # detach hat_m_ij during routing iteration to prevent gradient from flowing
        tmp_hat_q_ij = hat_q_ij.detach()
        # logger.info("hat_q_ij : {}".format(tmp_hat_q_ij.shape))
        print("tmp_hat_q_ij - is_leaf",tmp_hat_q_ij.is_leaf)

        a = torch.zeros(m.shape[0], self.num_caps, self.in_caps, 1).to(self.device)
        # checking('a', a)
        print("a - is_leaf",a.is_leaf)

        # logger.info(a.shape)

        # pearson corr
        # pearson = pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)
        # checking('pearson', pearson)

        p_ij = torch.tanh(pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)) # Change negative way
        # logger.info(p_ij.shape)
        # checking('p_ij', p_ij)


        # Dynamic Memory Routing Algorithm
        for routing_iter in range(self.num_routing-1):
            d = F.softmax(a, dim=1) # .to(self.device)
            # checking('d', d, True)
            print("\td - is_leaf",d.is_leaf)

            d_sum_p = d + p_ij
            # checking('d_sum_p', d_sum_p, True)
            print("\td_sum_p - is_leaf",d_sum_p.is_leaf)

            
#[ TODO ] : 여기 hat_m_cos_v 관련 뭔가 빠트린 것 같은데 추가할 것

            # Batch dimension에서 memory는 broadcast(expand) 되었고, sample vector q 는 n_class * k_samples를 나타내므로
            # Batch dimension은 sample vector q 기준 n_class * k_samples 개의 개별 벡터가 됌
            # [n_class * k_samples, num_caps, dim_caps]

            hat_v = (d_sum_p * tmp_hat_m_ij).sum(dim=2) 
            # checking('hat_v', hat_v, True)
            print("\that_v - is_leaf",hat_v.is_leaf)

            # [n_class * k_samples, num_caps, dim_caps]
            v = squash(hat_v)
            # checking('v', v, True)
            print("\tv - is_leaf", v.is_leaf)

            # dot product agreement between the v_j and the m_ij
            hat_m_cos_v = torch.matmul(tmp_hat_m_ij, v.unsqueeze(-1))
            # checking('hat_m_cos_v', hat_m_cos_v, True)
            print("\that_m_cos_v - is_leaf", hat_m_cos_v.is_leaf)

            a = a + torch.mul(tmp_hat_q_ij, hat_m_cos_v)
            # checking('a - after update', a, True)
            print("\ta - is_leaf", a.is_leaf)

            # [n_class*k_samples, num_caps, self.in_caps(n_memory), dim_caps]
            # copy (expand) to self.in_caps dim wise
            v = v.unsqueeze(2)
            v = v.expand(q.shape[0], self.num_caps, self.in_caps, self.dim_caps)
            # checking('v - after reshape', v, True)
            print("\tv expand - is_leaf", a.is_leaf)

            tmp_hat_q_ij = torch.div(torch.add(tmp_hat_q_ij, v), 2)
            # checking('tmp_hat_q_ij', tmp_hat_q_ij, True)
            print("\ttmp_hat_q_ij  - is_leaf", tmp_hat_q_ij.is_leaf)

            # tmp_hat_q_ij = torch.div(torch.add(tmp_hat_q_ij, v), self.n_class)
            p_ij = torch.tanh(pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)) # @@@@
            # checking('p_ij', p_ij, True)
            print("\tp_ij  - is_leaf", p_ij.is_leaf)


        # after routing ....
        d = F.softmax(a, dim=1) # .to(self.device)
        print("!d  - is_leaf", d.is_leaf)
        d_sum_p = d + p_ij
        print("!d_sum_p  - is_leaf", d_sum_p.is_leaf)
        print("!hat_m_ij  - is_leaf", hat_m_ij.is_leaf)
        print("!hat_m_ij  - requires_grad", hat_m_ij.requires_grad)

        hat_v = (d_sum_p * hat_m_ij).sum(dim=2) 
        print("hat_v - is leaf ", hat_v.is_leaf)
        v_out = squash(hat_v)
        print("v_out - is leaf ", v_out.is_leaf)
        print("!v_out  - is_leaf", v_out.is_leaf)

        # hat_m_cos_v = torch.matmul(tmp_hat_m_ij, v.unsqueeze(-1)) # ?
        # a = a + torch.mul(hat_q_ij, hat_m_cos_v)
        # v = v.unsqueeze(2)
        # v = v.expand(q.shape[0], self.num_caps, self.in_caps, self.dim_caps)
        # hat_q_ij = torch.div(torch.add(tmp_hat_q_ij, v), 2)
        # p_ij = torch.tanh(-pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)) # @@@@


        # after routing ....
        # v_out = v[:, :, 0, :] * self.in_caps
        # checking('v_out', v_out)
        # logger.info("v_out : {}".format(v_out.shape))
        v_out.requires_grad = True
        q_prime = v_out.view(q.shape[0], -1) # ???
        print("!q_prime  - is_leaf", q_prime.is_leaf)

        # checking('q_prime', q_prime)

        # logger.info(q_prime.shape)

        return q_prime

class DynamicTaskMemoryInduction(nn.Module):
    def __init__(self, in_caps, in_dim, 
                       num_caps, dim_caps,
                       num_routing,
                       W_share,
                       n_class, k_samples, device:torch.device):
        super(DynamicTaskMemoryInduction, self).__init__()
        self.in_dim = in_dim  # input capsule dimensionality (i.e. length)
        self.in_caps = in_caps
        self.dim_caps = dim_caps
        self.num_caps = num_caps # the number of high-level capsules == n_class 
        self.num_routing = num_routing

        # N = n_class, K = k_samples 
        # M = in_caps # the number of memories
        self.n_class = n_class
        self.k_samples = k_samples # QIM for n_querys
        self.n_support = self.n_class*self.k_samples
        # self.origin_memory == origin_memory
        # if self.origin_memory == True:
        #     self.n_memory = 

        # self.n_memory = 

        self.device = device

        # in_caps가 memory를 받을 때와 Query를 받을 때가 달라지므로 (X) 같음, memory는 support set의 개수
        # 가중치 공유를 위해 1로 두고 in_caps를 각각 다르게 주어 별도의 오브젝트로 만듦
        # self.W_share = nn.Parameter(0.01 * torch.randn(1, self.num_caps, 1, self.dim_caps, self.in_dim))
        self.W_share = 0.01 * torch.randn(1, self.num_caps, 1, self.dim_caps, self.in_dim)
        self.W = nn.Parameter(self.W_share.expand(1, self.num_caps, self.in_caps, self.dim_caps, self.in_dim).contiguous())
        # self.W = nn.Parameter(self.W_share.expand(1, self.num_caps, self.in_caps, self.dim_caps, self.in_dim).contiguous())
        self.b = nn.Parameter(torch.ones(1, self.num_caps, self.in_caps, self.dim_caps))
        self.register_parameter("DTMR_Wj", self.W)
        self.register_parameter("DTMR_Wj_bias", self.b)
        # self.tanh = torch.nn.Tanh

    def forward(self, m, q):
        '''
        m : W_base, memory [N_of_memory, d_dim=768]
        q : sample vector to be adapted to W_base [N_class*K_samples, d_dim=768]
        '''
        # (N*K (or M), H)
        # m = m.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(q.shape[0], self.num_caps, self.in_caps, self.dim_caps, 1).detach().clone()
        m = m.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(q.shape[0], 1, self.in_caps, self.in_dim, 1).contiguous() # / self.in_dim
        print("m ",m.is_contiguous())
        print("m - is_leaf",m.is_leaf)
        m.requires_grad = True

        # checking('m', m)
        # m = m.unsqueeze(0).unsqueeze(0).expand(m.shape[0], self.in_caps, self.in_dim).detach()
        
        # Initialize
        # W @ x =
        # (1, self.num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, self.num_caps, in_caps, dim_caps, 1)
        # logger.info("m.shape : {}".format(m.shape))
        # logger.info("W.shapt : {}".format(self.W.shape))
        # hat_m_ij = torch.matmul(self.W, m)
        print("self.W - is_leaf",self.W.is_leaf)
        print("self.W_share - is_leaf",self.W_share.is_leaf)
        print("self.b - is_leaf",self.b.is_leaf)

        hat_m_ij_ = torch.matmul(self.W, m).squeeze(-1)
        hat_m_ij_.requires_grad = True
        hat_m_ij = torch.add(hat_m_ij_, self.b)
        print(hat_m_ij.is_contiguous())
        print("!!hat_m_ij : ", hat_m_ij.requires_grad)
        print("hat_m_ij - is_leaf",hat_m_ij.is_leaf)
        print("self.W - is_leaf",self.W.is_leaf)
        # checking('W', self.W)
        # checking('b', self.b)
        checking('hat_m_ij', hat_m_ij)

        # logger.info(hat_m_ij.shape)

        # hat_m_ij = hat_m_ij.squeeze(-1)
        # checking('hat_m_ij', hat_m_ij)
        print("hat_m_ij : ", hat_m_ij.is_contiguous())

        # checking('hat_m_ij', hat_m_ij)

        # logger.info(hat_m_ij.shape)
        # detach hat_m_ij during routing iteration to prevent gradient from flowing
        tmp_hat_m_ij = hat_m_ij.detach()
        # logger.info("hat_m_ij : {}".format(tmp_hat_m_ij.shape))
        print("tmp_hat_m_ij - is_leaf",tmp_hat_m_ij.is_leaf)

        # Sample vector q s 
        # (n_class, k_samples, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        print("q before reshape - is_leaf",q.is_leaf)

        q_ = q.reshape(-1, 1, self.in_dim, 1)
        # q = q.reshape(q.shape[0], 1, self.in_dim, 1)
        print("q : ", q_.is_contiguous())
        print("q - is_leaf",q_.is_leaf)

        # logger.info(q.shape)
        # each sample vectors are copied to each memory but it should have diffrent memory address
        # when check with id function, it shows same id but
        # when use repeat, if you change the one of copies, they are dependent to each other
        # Above code, for memory, m is imutable value so, we don't need to use repeat to that. 
        # -> 하지만 다시 expand를 사용한다. 어차피 q_ij에서 i 즉 input_capsule(memory 개수)는
        #    계산을 위해 값이 차원따라서 반복만 될 뿐 같은 값이 되어야함
        # q = q.unsqueeze(2).repeat(1, 1, in_caps, 1, 1)
        # 수정함 : 들어오는 batch size에 맞게 expand 시켜야하므로.
        q__ = q_.unsqueeze(2).expand(q.shape[0], 1, self.in_caps, self.in_dim, 1) # Dont use this
        q__.requires_grad = True
        print("q : ", q__.is_contiguous())
        print("q - after expand - is_leaf",q.is_leaf)

        # logger.info(q.shape)
        # checking('q', m)

        # W @ x =
        # (1, self.num_caps, self.in_caps, dim_caps, in_dim) @ (batch_size, 1, self.in_caps, in_dim, 1) =
        # (batch_size, 1, self.num_caps, self.in_caps, dim_caps, 1)
        hat_q_ij = torch.matmul(self.W, q__)
        print("hat_q_ij : ", hat_q_ij.is_contiguous())
        # logger.info(hat_q_ij.shape)
        print("hat_q_ij - is_leaf",hat_q_ij.is_leaf)
        print("W - is_leaf",self.W.is_leaf)


        hat_q_ij = hat_q_ij.squeeze(-1)
        # logger.info(hat_q_ij.shape)
        # checking('hat_q_ij', hat_q_ij)
        print("hat_q_ij : ", hat_q_ij.is_contiguous())
        print("hat_q_ij squee - is_leaf",hat_q_ij.is_leaf)

        # detach hat_m_ij during routing iteration to prevent gradient from flowing
        tmp_hat_q_ij = hat_q_ij.detach()
        # logger.info("hat_q_ij : {}".format(tmp_hat_q_ij.shape))
        print("tmp_hat_q_ij - is_leaf",tmp_hat_q_ij.is_leaf)

        a = torch.zeros(m.shape[0], self.num_caps, self.in_caps, 1).to(self.device)
        # checking('a', a)
        print("a - is_leaf",a.is_leaf)

        # logger.info(a.shape)

        # pearson corr
        # pearson = pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)
        # checking('pearson', pearson)

        p_ij = torch.tanh(-pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)) # Change negative way
        # p_ij = torch.tanh(-pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)) # Change negative way
        print("p_ij - is_leaf",p_ij.is_leaf)

        # logger.info(p_ij.shape)
        # checking('p_ij', p_ij)


        # Dynamic Memory Routing Algorithm
        for routing_iter in range(self.num_routing-1):
            d = F.softmax(a, dim=1) # .to(self.device)
            # checking('d', d, True)
            print("\td - is_leaf",d.is_leaf)

            d_sum_p = d + p_ij
            # checking('d_sum_p', d_sum_p, True)
            print("\td_sum_p - is_leaf",d_sum_p.is_leaf)

            # Batch dimension에서 memory는 broadcast(expand) 되었고, sample vector q 는 n_class * k_samples를 나타내므로
            # Batch dimension은 sample vector q 기준 n_class * k_samples 개의 개별 벡터가 됌
            # [n_class * k_samples, num_caps, dim_caps]

            hat_v = (d_sum_p * tmp_hat_m_ij).sum(dim=2) 
            # checking('hat_v', hat_v, True)
            print("\that_v - is_leaf",hat_v.is_leaf)

            # [n_class * k_samples, num_caps, dim_caps]
            v = squash(hat_v)
            # checking('v', v, True)
            print("\tv - is_leaf", v.is_leaf)

            # dot product agreement between the v_j and the m_ij
            hat_m_cos_v = torch.matmul(tmp_hat_m_ij, v.unsqueeze(-1))
            # checking('hat_m_cos_v', hat_m_cos_v, True)
            print("\that_m_cos_v - is_leaf", hat_m_cos_v.is_leaf)
            
            # 세상에... 여기 잘못 되있었네.. hat_q_ij -> p_ij
            a = a + torch.mul(p_ij, hat_m_cos_v)
            # checking('a - after update', a, True)
            print("\ta - is_leaf", a.is_leaf)

            # [n_class*k_samples, num_caps, self.in_caps(n_memory), dim_caps]
            # copy (expand) to self.in_caps dim wise
            v = v.unsqueeze(2)
            v = v.expand(q.shape[0], self.num_caps, self.in_caps, self.dim_caps)
            # checking('v - after reshape', v, True)
            print("\tv expand - is_leaf", a.is_leaf)

            tmp_hat_q_ij = torch.div(torch.add(tmp_hat_q_ij, v), 2)
            # checking('tmp_hat_q_ij', tmp_hat_q_ij, True)
            print("\ttmp_hat_q_ij  - is_leaf", tmp_hat_q_ij.is_leaf)

            # tmp_hat_q_ij = torch.div(torch.add(tmp_hat_q_ij, v), self.n_class)
            p_ij = torch.tanh(-pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)) # @@@@
            # checking('p_ij', p_ij, True)
            print("\tp_ij  - is_leaf", p_ij.is_leaf)

        # after routing ....
        d = F.softmax(a, dim=1) # .to(self.device)
        print("!d  - is_leaf", d.is_leaf)

        d_sum_p = d + p_ij
        print("!d_sum_p  - is_leaf", d_sum_p.is_leaf)

        hat_v = (d_sum_p * hat_m_ij).sum(dim=2) 
        print("!hat_v  - is_leaf", hat_v.is_leaf)
        print("!hat_m_ij  - is_leaf", hat_m_ij.is_leaf)
        print("!hat_m_ij  - requires_grad", hat_m_ij.requires_grad)
        v_out = squash(hat_v)
        print("!v_out  - is_leaf", v_out.is_leaf)
        # hat_m_cos_v = torch.matmul(tmp_hat_m_ij, v.unsqueeze(-1)) # ?
        # a = a + torch.mul(hat_q_ij, hat_m_cos_v)
        # v = v.unsqueeze(2)
        # v = v.expand(q.shape[0], self.num_caps, self.in_caps, self.dim_caps)
        # hat_q_ij = torch.div(torch.add(tmp_hat_q_ij, v), 2)
        # p_ij = torch.tanh(-pearsonr(tmp_hat_m_ij, tmp_hat_q_ij)) # @@@@


        # v_out = v[:, :, 0, :] * self.in_caps # for backpropagation
        # checking('v_out', v_out)
        # logger.info("v_out : {}".format(v_out.shape))
        v_out.requires_grad = True
        q_prime = v_out.view(q.shape[0], -1)
        print("!q_prime  - is_leaf", q_prime.is_leaf)
        # q_prime = v_out.reshape(q.shape[0], -1)
        # checking('q_prime', q_prime)

        # logger.info(q_prime.shape)

        return q_prime

class SimilarityClassifier(nn.Module):
    def __init__(self, n_class, n_query, hidden_dim, device):
        super(SimilarityClassifier, self).__init__()
        self.n_class = n_class
        self.n_query = n_query 
        self.n_samples = self.n_class * self.n_query
        self.hidden_dim = hidden_dim
        self.device = device
        
        self.tau = torch.nn.Parameter(0.01 * torch.ones(1))
        self.register_parameter("Sim_tau", self.tau)
        # self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, e_c, e_q):  # (N_class*(N_class*N_query),H) (N_class*N_query,H) ?
        # [TODO] N_query는 입력에 따라 유동적이게 바꿔야함
        # 각각의 Query에 대해서 2개씩 Claass vector를 받게됨

        # print("e_c 1 ", e_c.shape)
        print("e_c in ", e_c.is_contiguous())
        # checking('e_c in', e_c)
        
        e_c_norm = F.normalize(e_c, p=2, dim=-1).view(self.n_class, e_q.shape[0], self.hidden_dim)
        # checking('e_c in', e_c)
        e_c_norm.requires_grad = True

        print("e_c contiguous", e_c.is_contiguous())
        print("e_c_norm  - is contiguous", e_c_norm.is_contiguous)
        print("e_c_norm  - is leaf", e_c_norm.is_leaf)
        # e_c = F.normalize(e_c, p=2, dim=-1)
        # e_c_norm = torch.norm(e_c, p=2, dim=-1).detach()
        # e_c = e_c.div(e_c_norm.unsqueeze(-1).expand_as(e_c))
        # print("e_c 2 ", e_c.shape)
        # e_c = e_c.view(self.n_class, e_q.shape[0], self.hidden_dim)
        # checking('e_c in', e_c)


        # e_q_norm = torch.norm(e_q, p=2, dim=-1).detach()
        # e_q = e_q.div(e_q_norm.unsqueeze(-1).expand_as(e_q))
        e_q_norm = F.normalize(e_q, p=2, dim=-1)
        # checking('e_q in', e_q)
        print("e_q_norm  - is leaf", e_q_norm.is_leaf)

        # logger.info("e_c : {}".format(e_c.shape))
        # logger.info("e_q : {}".format(e_c.shape))
        
        # (N_class, N_class * N_query, H),  (N_class * N_query, H)
        # s_qc_b = F.cosine_similarity(e_c, e_q, dim=-1)
        s_qc_b = torch.sum(torch.mul(e_c_norm, e_q_norm), -1).T.contiguous()
        s_qc_b.requires_grad = True
        print("s_qc_b", s_qc_b.is_contiguous())
        print("s_qc_b  - is leaf", s_qc_b.is_leaf)

        # (N_class, N_class * N_query)        
        # logger.info("s_qc_b : {}".format(s_qc_b.shape))
        s_qc = torch.mul(self.tau, s_qc_b).squeeze(-1)
        print("s_qc  - is leaf", s_qc.is_leaf)
        print("s_qc  - requires_grad", s_qc.requires_grad)
        print("s_qc  - is contiguous", s_qc.is_contiguous())
        # (N_class * N_query, N_class)        
        # probs = F.softmax(s_qc.T,dim=1)
        # s_qc = s_qc.T
        return s_qc

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
    def __init__(self, n_class, k_samples, n_query,
                 memory, n_routing, device):
        super(DMRInduction, self).__init__()

        # Params of N-way K-shot setting
        self.n_class = n_class
        self.k_samples = k_samples
        self.n_support = self.n_class * self.k_samples
        self.n_query = n_query
        self.device = device
        # Parames of Encoder 
        self.encoder = BertModel.from_pretrained('bert-base-uncased') # path?
        # To check the bert input, See the Sandbox3.ipynb in google drive
        # Freeze the BERT
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.to(self.device)
        
        # Parames of Capsule Networks
        self.memory = memory.detach()
        self.n_memory, self.hidden_dim = memory.shape # (N_memory, hidden_dim=768) 
        self.dim_caps = self.hidden_dim // self.n_class
        self.n_routing = n_routing
        # self.W_share = nn.Parameter(0.01 * torch.randn(1, self.n_class, 1, self.dim_caps, self.hidden_dim)).to(self.device)
        # self.W_share2 = nn.Parameter(0.01 * torch.randn(1, self.n_class, 1, self.dim_caps, self.hidden_dim)).to(self.device)
        # self.W_share2 = nn.Parameter(0.01 * torch.randn(1, self.n_class, 1, self.dim_caps, self.hidden_dim)).to(self.device)


        self.TaskRetrieval = TaskRetrival(self.memory, top_k=1)

        self.DTMR = DynamicTaskMemoryInduction(
             self.TaskRetrieval.n_memory, self.hidden_dim, self.n_class,
            #  self.n_memory, self.hidden_dim, self.n_class,
             self.dim_caps, self.n_routing, None, self.n_class, self.k_samples,  self.device).to(self.device)
        
        # It differ with DMR at the number of the input capsules
        self.QIM = DynamicMemoryInduction(
             self.k_samples, self.hidden_dim, self.n_class,
             self.dim_caps, self.n_routing, None, self.n_class, self.n_query, self.device).to(self.device)

        self.simclassifier = SimilarityClassifier(self.n_class, self.n_query, self.hidden_dim, self.device).to(self.device)

    def forward(self, data, attmask, segid):
        # (batch[support-5 + query-27], hidden_size=768)
        pooler_output  = self.encoder(data, attmask, segid)[1]
        print("pooler_output : ", pooler_output.requires_grad)
        print("pooler_output is_leaf: ", pooler_output.is_leaf)

        # [N_class*K_samples, in_dim], [N_class*N_querys, in_dim]
        # print("!!!pooler_output : ", pooler_output.shape)
        # print(torch.isnan(pooler_output).sum())
        # print(torch.isinf(pooler_output).sum())

        support_encoder, query_encoder = pooler_output[0:self.n_support], pooler_output[self.n_support:]
        # print("S ", support_encoder.shape)
        # print("Q ",query_encoder.shape)
        # Dynamic Memory Routing Process
        # Input :
        #   - m : W_base, memory [N_of_memory, d_dim=768]
        #   - q : sample vector to be adapted to W_base [N_class*K_samples, d_dim=768]
        # Output :
        #   - e_cs : [N_class * K_samples, in_dim]
        print("support_encoder is_leaf: ", support_encoder.is_leaf)
        print("query_encoder is_leaf: ", query_encoder.is_leaf)

        ret_memory = self.TaskRetrieval(support_encoder)
        print("ret_memory is_leaf: ", ret_memory.is_leaf)

        e_cs = self.DTMR(ret_memory, support_encoder)
        print("e_cs is_leaf: ", e_cs.is_leaf)

        # Query-enhanced Induction Module
        # Batch Size 쪽에서 불일치 발생가능성
        # Input :
        #   - e_cs : adapted sample vectors
        #   - query_encoder : sample vector to be adapted to q_prime [N_class*N_querys, d_dim=768]
        # Output :
        #   - e_c# : [N_class*N_querys, in_dim]
        # -> query vector 마다 N_class 개수 만큼의 e_c를 얻음
        
        # [TODO] Solve the Hard coding 

        e_cs = e_cs.view(self.n_class, self.k_samples, self.hidden_dim)
        # e_cs = e_cs.reshape(self.n_class, self.k_samples, self.hidden_dim)
        print("e_cs is_leaf: ", e_cs.is_leaf)

        # print("!!!DMR_output reshape")
        # checking("DMR_output", e_cs)

        e_c0, e_c1 = e_cs[0, :, :], e_cs[1, :, :]
        # checking("e_c0", e_c0)
        # checking("e_c1", e_c1)

        e_c0 = self.QIM(e_c0, query_encoder)
        e_c1 = self.QIM(e_c1, query_encoder)
        print("e_c0 ", e_c0.is_contiguous())
        print("e_c1 ", e_c1.is_contiguous())
        print("e_c0 is_leaf: ", e_c0.is_leaf)
        print("e_c1 is_leaf: ", e_c1.is_leaf)

        # print('after QIM-----------')

        # checking("e_c0", e_c0)
        # checking("e_c1", e_c1)

        # logger.info("e_c0 : {}".format(e_c0.shape))
        # logger.info("e_c1 : {}".format(e_c1.shape))
        
        # [N_class, N_class*N_querys, in_dim]
        e_c = torch.cat([e_c0, e_c1], dim=0)
        print("e_c  ", e_c.is_contiguous())
        print("e_c is_leaf: ", e_c.is_leaf)
        
        logits = self.simclassifier(e_c, query_encoder)
        print("logits : ", logits.requires_grad)
        print("logits is_leaf: ", logits.is_leaf)
        # checking("probs", e_c0)
        return logits
