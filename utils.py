import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.lines import Line2D
import numpy as np
from torch.nn.modules.loss import _Loss

class Criterion(_Loss):
    def __init__(self, way=2, shot=5):
        super(Criterion, self).__init__()
        # self.amount = way * shot

    def forward(self, probs, target):  # (Q,C) (Q)
        # target = target[self.amount:]
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1).unsqueeze(-1)
        # print("="*50)
        # print("target & pred")
        # print(target)
        # print(pred)
        # print("="*50)

        acc = torch.sum(target == pred).float() / target.shape[0]
        return loss, acc

def get_accuracy(prob, labels):
    index = np.array(torch.argmax(prob.detach().cpu(), dim=1).numpy()).tolist()
    labels = np.array(torch.argmax(labels.detach().cpu(), dim=1).numpy()).tolist()
    print(index)
    length = len(labels)
    num = 0
    for i in range(length):
        if index[i] == labels[i]:
            num += 1

    return num/length


def label2tensor(label, device, n_class = 2):
#     label_dict = {}
#     for index, _label in enumerate(set(label)):
#         label_dict[_label] = index

#     label2id = torch.LongTensor([[label_dict[k]] for k in label])
    
    label_one_hot = torch.zeros(len(label), n_class, device=device).scatter_(1, label, 1)

    return label_one_hot


def checking(name, m, indent=False):
    pass

# def checking(name, m, indent=False):
#     if indent:
#         print("\t{} - shape {}".format(name, m.shape))
#         print("\t{} - is nan {}".format(name, torch.isnan(m).sum()))
#         print("\t{} - is inf {}".format(name, torch.isinf(m).sum()))
#         print("\t{} - max : {} / min : {} ".format(name, torch.max(m),torch.min(m)))
#         print("\t{} - reuires_grad {}".format(name, m.requires_grad))
#         print("\t{} - is_leaf {}".format(name, m.is_leaf))
#     else:
#         print("{} - shape {}".format(name, m.shape))
#         print("{} - is nan {}".format(name, torch.isnan(m).sum()))
#         print("{} - is inf {}".format(name, torch.isinf(m).sum()))
#         print("{} - max : {} / min : {} ".format(name, torch.max(m),torch.min(m)))
#         print("{} - reuires_grad {}".format(name, m.requires_grad))
#         print("{} - is_leaf {}".format(name, m.is_leaf))


def param_record(epoch, recored_interval, model_parameters, writer, n_iter):
    if epoch % recored_interval == 0:
        for name, param in model_parameters.items():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)

def embed_record(support_encoder, query_encoder, ret_memory, e_cs, e_cq, e_c_end):
    support_encoder = support_encoder.detach().cpu().numpy()
    query_encoder = query_encoder.detach().cpu().numpy()
    ret_memory = ret_memory.detach().cpu().numpy()
    e_cs = e_cs.detach().cpu().numpy()
    e_cq = e_cq.detach().cpu().numpy()
    #####
    e_c_end = e_c_end.detach().cpu().numpy()

    pca_enc = PCA(2)
    pca_e_csq = PCA(2)
    pca_e_c_end = PCA(2)

    s_q_r = np.concatenate((support_encoder, query_encoder, ret_memory))
    s_q_r_sc = sc_enc.fit_transform(s_q_r)
    s_q_r_pca = pca_enc.fit_transform(s_q_r_sc)
    # Support class 0, 1
    plt.scatter(s_q_r_pca[:5, 0], s_q_r_pca[:5, 1], c='r')
    plt.scatter(s_q_r_pca[5:10, 0], s_q_r_pca[5:10, 1], c='y')
    # Query class 0, 1
    plt.scatter(s_q_r_pca[10:37, 0], s_q_r_pca[10:37, 1], c='b')
    plt.scatter(s_q_r_pca[37:64, 0], s_q_r_pca[37:64, 1], c='c')
    # Retrived Task Mean Vector
    plt.scatter(s_q_r_pca[64:, 0], s_q_r_pca[64:, 1], c='g')


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.01) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('fig2.png', dpi=100)

# def plot_grad_flow(named_parameters):
#     ave_grads = []
#     layers = []
#     for n, p in named_parameters:
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#     plt.plot(ave_grads, alpha=0.3, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(xmin=0, xmax=len(ave_grads))
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.savefig('fig1.png', dpi=300)

def accuracy(predict, target):
    _, predicted = torch.max(predict.data, 1)
    correct = (predicted == target).sum().item()
    return correct / len(target)

def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:  
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")
    print("\n")

def padding(data1, data2, pad_idx=0):
    len1, len2 = data1.shape[1], data2.shape[1]
    if len1 > len2:
        data2 = torch.cat([data2, torch.ones(data2.shape[0], len1 - len2).long() * pad_idx], dim=1)
    elif len2 > len1:
        data1 = torch.cat([data1, torch.ones(data1.shape[0], len2 - len1).long() * pad_idx], dim=1)
    return data1, data2


def batch_padding_bertinput(data, pad_idx=0):
    max_len = 0
    attmasks = []
    seg_ids = []

    for text in data:
        max_len = max(max_len, len(text))
    for i in range(len(data)):
        data[i] += [pad_idx] * (max_len - len(data[i]))
        attmask = [1] * len(data[i]) + [0] * (max_len - len(data[i]))
        seg_id = [1] * len(data[i]) + [0] * (max_len - len(data[i]))
        attmasks.append(attmask)
        seg_ids.append(seg_id)

    return torch.tensor(data), torch.tensor(attmasks),  torch.tensor(seg_ids) 


def squash(s, dim=-1):
    '''
    "Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
    Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||

    Args:
    s: 	Vector before activation
    dim:	Dimension along which to calculate the norm

    Returns:
    Squashed vector
    '''
    # print("s : ", s)

    # squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
    squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
    # print("squared_norm : ", squared_norm)
    # print("squared_norm : ", torch.sqrt(squared_norm)+ 1e-8)
    # print("squared_norm : ", 1/(1 + squared_norm))

    # print("squared_norm : ", torch.isnan(torch.sqrt(squared_norm)+ 1e-8).sum())
    # print("squared_norm : ", torch.isnan((squared_norm / (1 + squared_norm))).sum())
    # print("squared_norm : ", (squared_norm / (1 + squared_norm)))

    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm)+ 1e-8)

def pearsonr(x, y, batch_first=True):
    r"""Computes Pearson Correlation Coefficient across rows.
    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:
    .. math::
        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}
    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.
    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`
    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`
    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:
        .. math::
            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2
        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.
    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors
    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])
    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr

