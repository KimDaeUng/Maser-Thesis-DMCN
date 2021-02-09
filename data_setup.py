import configparser
import os
import re
import string
import pickle
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from dataloader import TrainDataLoader
from utils import padding, batch_padding_bertinput
from transformers import BertTokenizer


def _parse_list(data_path, list_name):
    domain = set()
    with open(os.path.join(data_path, list_name), 'r', encoding='utf-8') as f:
        for line in f:
            domain.add(line.strip('\n'))
    return domain


def get_domains(data_path, filtered_name, target_name):
    all_domains = _parse_list(data_path, filtered_name)
    test_domains = _parse_list(data_path, target_name)
    train_domains = all_domains - test_domains
    print('train domains', len(train_domains), 'test_domains', len(test_domains))
    return sorted(list(train_domains)), sorted(list(test_domains))


def _parse_data(data_path, filename):
    neg = {
        'filename': filename,
        'data': [],
        'target': []
    }
    pos = {
        'filename': filename,
        'data': [],
        'target': []
    }
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if line[-2:] == '-1':
                neg['data'].append(line[:-2])
                neg['target'].append(0)
            else:
                pos['data'].append(line[:-1])
                pos['target'].append(1)
    # check
    print(filename, 'neg', len(neg['data']), 'pos', len(pos['data']))
    return neg, pos


def _process_data(data_dict):
    for i in range(len(data_dict['data'])):
        text = data_dict['data'][i]
        # ignore string.punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        # string.whitespace -> space
        text = re.sub('[%s]' % re.escape(string.whitespace), ' ', text)
        # lower case
        text = text.lower()
        # split by whitespace
        text = text.split()
        # replace
        data_dict['data'][i] = text
    return data_dict


def _get_data(data_path, domains, usage):
    # usage in ['train', 'dev', 'test']
    data = {}
    for domain in domains:
        for t in ['t2', 't4', 't5']:
            filename = '.'.join([domain, t, usage])
            neg, pos = _parse_data(data_path, filename)
            neg = _process_data(neg)
            pos = _process_data(pos)
            data[filename] = {'neg': neg, 'pos': pos}
    return data


def get_train_data(data_path, domains):
    train_data = _get_data(data_path, domains, 'train')
    print('train data', len(train_data))
    return train_data


def _combine_data(support_data, data):
    # support -> dev, test
    for key in data:
        key_split = key.split('.')[0:-1] + ['train']
        support_key = '.'.join(key_split)
        for value in data[key]:
            data[key][value]['support_data'] = copy.deepcopy(support_data[support_key][value]['data'])
            data[key][value]['support_target'] = copy.deepcopy(support_data[support_key][value]['target'])
    return data


def get_test_data(data_path, domains):
    # get dev, test data
    support_data = _get_data(data_path, domains, 'train')
    dev_data = _get_data(data_path, domains, 'dev')
    test_data = _get_data(data_path, domains, 'test')

    # support -> dev, test
    dev_data = _combine_data(support_data, dev_data)
    test_data = _combine_data(support_data, test_data)
    print('dev data', len(dev_data), 'test data', len(test_data))
    return dev_data, test_data

def _idx_text(text_list, tokenizer):
    for i in range(len(text_list)):
        text_list[i] = tokenizer.encode(text_list[i], max_length=512, truncation=True, padding=True)
    return text_list

def idx_all_data(data, tokenizer):
    for filename in data:
        print(filename)
        for value in data[filename]:
            for key in data[filename][value]:
                if key in ['data', 'support_data']:
                    data[filename][value][key] = _idx_text(data[filename][value][key], tokenizer)
    return data


def get_train_loader(train_data, support, query, pad_idx):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    batch_size = support + query
    train_loaders = {}
    for filename in train_data:
        neg_dl = DataLoader(Dataset(train_data[filename]['neg'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        pos_dl = DataLoader(Dataset(train_data[filename]['pos'], pad_idx), batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        if min(len(neg_dl), len(pos_dl)) > 0:
            train_loaders[filename] = {
                'neg': neg_dl,
                'pos': pos_dl
            }
    print('train loaders', len(train_loaders))
    return TrainDataLoader(train_loaders, support=support, query=query, pad_idx=pad_idx)


def get_test_loader(full_data, support, query, pad_idx):
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    loader = []
    for filename in full_data:
        # support
        support_data = full_data[filename]['neg']['support_data'][0:support] + full_data[filename]['pos']['support_data'][0:support]
        support_data, support_attmask, support_segid = batch_padding_bertinput(support_data, pad_idx)
        support_target = full_data[filename]['neg']['support_target'][0:support] + full_data[filename]['pos']['support_target'][0:support]
        support_target = torch.tensor(support_target)
        # query
        neg_dl = DataLoader(Dataset(full_data[filename]['neg'], pad_idx), batch_size=query * 2, shuffle=False, drop_last=False, **kwargs)
        pos_dl = DataLoader(Dataset(full_data[filename]['pos'], pad_idx), batch_size=query * 2, shuffle=False, drop_last=False, **kwargs)
        # combine
        for dl in [neg_dl, pos_dl]:
            for batch_data, batch_attmask, batch_segid, batch_target in dl:
                support_data_cp, support_attmask_cp = copy.deepcopy(support_data), copy.deepcopy(support_attmask), 
                support_segid_cp, support_target_cp = copy.deepcopy(support_segid), copy.deepcopy(support_target)
                
                support_data_cp, batch_data = padding(support_data_cp, batch_data, pad_idx)
                support_attmask_cp, batch_attmask = padding(support_attmask_cp, batch_attmask, pad_idx)
                support_segid_cp, batch_segid = padding(support_segid_cp, batch_segid, pad_idx)

                data = torch.cat([support_data_cp, batch_data], dim=0)
                attmask = torch.cat([support_attmask_cp, batch_attmask], dim=0)
                segid = torch.cat([support_segid_cp, batch_segid], dim=0)
                target = torch.cat([support_target_cp, batch_target], dim=0)
                
                loader.append((data, attmask, segid, target))
    print('test loader length', len(loader))
    return loader


def main():
    train_domains, test_domains = get_domains(data_path_raw, config['Data']['filtered_list'], config['Data']['target_list'])
    
    train_data = get_train_data(data_path_raw, train_domains)
    dev_data, test_data = get_test_data(data_path_raw, test_domains)
    # print(dev_data['books.t2.dev']['neg']['support_data'])
    # print(dev_data['books.t2.dev']['neg']['support_target'])


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    pad_idx = tokenizer.pad_token_id

    train_data = idx_all_data(train_data, tokenizer)
    dev_data = idx_all_data(dev_data, tokenizer)
    test_data = idx_all_data(test_data, tokenizer)
    # print(dev_data['books.t2.dev']['neg']['support_data'])
    # print(dev_data['books.t2.dev']['neg']['support_target'])

    support = int(config['Model']['support'])
    query = int(config['Model']['query'])
    train_loader = get_train_loader(train_data, support, query, pad_idx)
    dev_loader = get_test_loader(dev_data, support, query, pad_idx)
    test_loader = get_test_loader(test_data, support, query, pad_idx)
    
    # Below files are in google drive 20.10.10.
    torch.save(train_loader, os.path.join(data_path_pickle, config['Data']['train_loader']))
    # pickle.dump(train_loader, open(os.path.join(config['Data']['path'], config['Data']['train_loader']), 'wb'))
    pickle.dump(dev_loader, open(os.path.join(data_path_pickle, config['Data']['dev_loader']), 'wb'))
    pickle.dump(test_loader, open(os.path.join(data_path_pickle, config['Data']['test_loader']), 'wb'))


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['Data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    data_path = config['Data']['path']
    data_path_raw = config['Data']['path_raw']
    data_path_pickle = config['Data']['path_pickle']


    main()