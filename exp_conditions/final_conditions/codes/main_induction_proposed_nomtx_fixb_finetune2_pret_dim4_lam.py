from configparser import ConfigParser, ExtendedInterpolation
import pickle
import os
import torch
import wandb
from torch import optim
import random
import numpy as np
from model_induction_proposed_nomtx_fixb_ftune2nd_pret import DMRInduction
from criterion import CosineLoss, log_softmax, cross_etrp #, Criterion
from tensorboardX import SummaryWriter
from transformers import BertModel
from dataloader import *
from dataset import Dataset
import tqdm 
# from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.lines import Line2D
from utils import Criterion, plot_grad_flow, accuracy, param_record, hook_fn, label2tensor, get_accuracy
import glob
import gc
import copy
def train(episode):
    model.train()
    data, attmask, segid, target = train_loader.get_batch()
    
    data = data.detach().to(device)
    attmask = attmask.detach().to(device)
    segid = segid.detach().to(device)

    # We only calcuate the loss of the query set 
    target = target[n_support:].to(device).unsqueeze(-1)
    
    _, predict = model(data, attmask, segid, episode) # (n_query, n_class)
    loss, acc = criterion(predict, target)
    
    optimizer.zero_grad()
    loss.backward()

    # plot_grad_flow(model.named_parameters())
    # param_record(episode, 1, model_parameters, writer, episode)
    # wandb.log({'gradients' : wandb.Histogram()})
    if episode % log_interval == 0:
        for name, param in model_parameters.items():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), episode)

    optimizer.step()


    writer.add_scalar('train_loss', loss.item(), episode)
    writer.add_scalar('train_acc', acc, episode)

    if episode % log_interval == 0:
        # wandb.log({'train_loss' : loss.item(), 'train_acc' : acc})

        print('Train Episode: {} Loss: {} Acc: {}'.format(episode, loss.item(), acc))
        # print("predict\n")
        # print(predict)
        # print("target\n")
        # print(target)
    


def dev(episode):
    model.eval()
    correct = 0.
    count = 0.

    
    for data, attmask, segid, target in tqdm.tqdm(dev_loader):
        data = data.to(device)
        attmask = attmask.to(device)
        segid = segid.to(device)
        # print(data.shape)
        # We only calcuate the loss of the query set 
        target = target[n_support:].to(device).unsqueeze(-1)

        _, predict = model(data, attmask, segid, episode) # (n_query, n_class)
        _, acc = criterion(predict, target)
        
        amount = len(target)
        correct += acc * amount
        count += amount
    
    acc = correct / count
    writer.add_scalar('dev_acc', acc, episode)
    # wandb.log({'dev_acc' : acc})
    print('Dev Episode: {} Acc: {}'.format(episode, acc))

    return acc


def test(epi):
    model.eval()
    correct = 0.
    count = 0.
    for data, attmask, segid, target in tqdm.tqdm(test_loader):
        data = data.to(device)
        attmask = attmask.to(device)
        segid = segid.to(device)
        # We only calcuate the loss of the query set 
        target = target[n_support:].to(device).unsqueeze(-1)

        _, predict = model(data, attmask, segid) # (n_query, n_class)
        _, acc = criterion(predict, target)
        
        amount = len(target)
        correct += acc * amount
        count += amount

    acc = correct / count
    writer.add_scalar('test_acc', acc, epi)
    # wandb.log({'test_acc' : acc})
    print('Test Acc: {}'.format(acc))
    return acc

def main():

    ##################################################

    print("Start Training")
    best_episode, best_acc = 0, 0.

    
    episodes = int(config['Model']['episodes'])
    early_stop = int(config['Model']['early_stop']) * dev_interval

    start_epi = 1


    if continue_train == 1:
        ckpt_path = os.path.join(config['Log']['ckpt_path'], 'S-5_Q-27_LR-1e-5_Tmp-1_TopK-10_NRouting-3_NCapsule2_CompDim-4_NLastLayer-2_TAG-induction_2_proposed_fixb_ftune2_pret_dim4_lam_Epi-13200.h5') 
        # list_of_ckpts = glob.glob(ckpt_path)
        # latest_file = max(list_of_ckpts, key=os.path.getctime)
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)
        del ckpt
        gc.collect()

        print("Load ckpts : {}".format(ckpt_path))
        start_epi = int(ckpt_path.split("Epi-")[-1][:-3])
        print("Continue From {} th episode, ".format(start_epi))

    for episode in tqdm.trange(start_epi, episodes + 1):
        train(episode)
        

        if (episode % dev_interval == 0):
            print('Start evaluation on Test set')
            # acc = dev(episode)
            acc = test(episode)
            print('{}th epicode acc : '.format(episode), acc)
            if acc > best_acc:
                print('Best Test ACC, Saving model!')
                model_ckpt_path = os.path.join(config['Log']['ckpt_path'],
                    "{}_Epi-{}.h5".format(config['Log']['name'], episode))
                torch.save(model.state_dict(), model_ckpt_path)
                # wandb.save("{}_Epi-{}.h5".format(config['Log']['name'], episode))
                best_episode, best_acc = episode, acc

            if episode - best_episode >= early_stop:
                print('Early stop at episode', episode)
                break
        
    print('Reload the best model on episode', best_episode, 'with best acc', best_acc)
    ckpt = torch.load(model_ckpt_path)
    model.load_state_dict(ckpt)
    print("Start testing on test set")
    test()


if __name__ == "__main__":
    # move bach dir to project dir
    config = ConfigParser(interpolation=ExtendedInterpolation())
    # config.read("config_2_proposed_comp_dim_less.ini")
    # config.read("config_2_proposed_comp_dim_lessless.ini")
    # config.read("config_2_proposed_comp_dim_6.ini")
    # config.read("config_2_proposed_comp_dim_8.ini")
    # config.read("config_2_proposed_comp_dim_5.ini")
    # config.read("config_2_proposed_comp_dim_3.ini")
    # config.read("config_2_proposed_comp_lambda.ini")
    # config.read("config_2_proposed_comp_lambdam.ini")
    # config.read("config_2_proposed_comp_nomtx.ini")
    # config.read("config_2_proposed_comp_nomtxseed.ini")
    # config.read("config_2_proposed_comp_nomtx_pret.ini")
    # config.read("config_2_proposed_comp_nomtx_fixb.ini")
    # config.read("config_2_proposed_comp_nomtx_fixb_ftune2.ini")
    # config.read("config_2_proposed_comp_nomtx_fixb_ftune2_pret.ini")
    # config.read("config_2_proposed_comp_nomtx_fixb_ftune2_pret_dim4.ini")
    config.read("config_2_proposed_comp_nomtx_fixb_ftune2_pret_dim4_lam.ini")
    # config.read("config_2_proposed_comp_dim_more.ini")
    print(config['Log']['name'])
    # wandb.init(project='final_cross', reinit=True, config=config._sections)

    # # config run name
    # wandb.run.name = config['Log']['tag']
    # wandb.run.save()

    # Offline
    # os.environ['WANDB_API_KEY'] = config['Wandb']['api_key']
    # os.environ['WANDB_MODE'] = config['Wandb']['mode']

    # log_interval
    log_interval = int(config['Log']['log_interval'])
    dev_interval = int(config['Log']['dev_interval'])

    # data loaders
    print("Load datasets")
    print(os.getcwd())
    print(os.path.join(config['Data']['path_pickle'], config['Data']['train_loader']))
    # print('-'*50)
    # encoder = BertModel.from_pretrained('tmp/finetuend_lm/', output_hidden_states=True) # path?
    # exit()

    # train_loader = None
    train_loader = torch.load(os.path.join(config['Data']['path_pickle'], config['Data']['train_loader']))
    # dev_loader = None
    dev_loader = pickle.load(open(os.path.join(config['Data']['path_pickle'], config['Data']['dev_loader']), 'rb'))
    test_loader = pickle.load(open(os.path.join(config['Data']['path_pickle'], config['Data']['test_loader']), 'rb'))

    # model & optimizer & criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_class = int(config['Model']['class'])
    support = int(config['Model']['support'])
    query = int(config['Model']['support'])
    n_support = n_class * support
    top_k = int(config['Model']['top_k'])
    n_routing = int(config['Model']['n_routing'])
    comp_dim = int(config['Model']['comp_dim'])

    last_layer = int(config['Model']['memory_last_layer'])

    continue_train = int(config["Log"]['continue'])


    # # Memory Matrix
    # if last_layer == 1:
    #     memory_mtx = torch.load(os.path.join(config['Data']['path_pickle'], 'meta_train_task_memory_new.pt')).to(device)
    # elif last_layer == 2:
    #     memory_mtx = torch.load(os.path.join(config['Data']['path_pickle'], 'meta_train_task_memory_last2.pt')).to(device)
    # else:
    #     # Random
    #     memory_mtx = torch.randn(57, 768).to(device)
    
    print('Build Model')
    # seed
    seed = int(config['Model']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = DMRInduction(n_class=n_class,
                             k_sample=support,
                             n_query=query,
                            #  memory=memory_mtx,
                             n_routing=n_routing,
                             top_k=top_k,
                             comp_dim=comp_dim,
                             device=device)
    # wandb.watch(model)

    model.to(device)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Requires_grad True parameters : ")
            print(name)

    model_parameters = { name : param for name, param in model.named_parameters() if param.requires_grad}

    # model.DTMR.register_backward_hook(hook_fn)
    # model.DTMR.b.register_backward_hook(hook_fn)
    # model.QIM.register_backward_hook(hook_fn)
    # model.QIM.b.register_backward_hook(hook_fn)
    # model.simclassifier.register_backward_hook(hook_fn)

    optimizer = optim.Adam(model.parameters(), lr=float(config['Model']['lr']))
    # criterion = CosineLoss(device)
    # criterion = cross_etrp
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = Criterion()


    # writer
    os.makedirs(config['Log']['log_tb_path'], exist_ok=True)
    os.makedirs(config['Log']['log_value_path'], exist_ok=True)
    os.makedirs(config['Log']['emb_path'], exist_ok=True)
    os.makedirs(config['Log']['ckpt_path'], exist_ok=True)
    print()
    writer = SummaryWriter(config['Log']['log_tb_path'])

    main()
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()