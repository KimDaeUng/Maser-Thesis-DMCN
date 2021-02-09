import copy
import torch
import torch.utils.data
from utils import padding


class TrainDataLoader:
    def __init__(self, loaders, support, query, pad_idx):
        # original
        self.loaders = loaders
        self.filenames = sorted(loaders.keys())
        self.loaders_ins = self.instantiate_all(loaders)
        # current indices
        self.index = -1
        self.indices = self.reset_indices(loaders)
        # max indices
        self.max_indices = self.get_batch_cnt(loaders)
        # arg
        self.support = support
        self.query = query
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.loaders)

    def instantiate_one(self, loader):
        return list(copy.deepcopy(loader))

    def instantiate_all(self, loader):
        new_loader = {}
        for filename in loader:
            new_loader[filename] = {}
            for value in loader[filename]:
                new_loader[filename][value] = self.instantiate_one(loader[filename][value])
        return new_loader # Batch * [data, attmask_segid, target]

    def reset_indices(self, loader):
        indices = {}
        for filename in loader:
            indices[filename] = {}
            for value in loader[filename]:
                indices[filename][value] = 0
        return indices

    def get_batch_cnt(self, loader):
        batch_cnt = {}
        for filename in loader:
            batch_cnt[filename] = {}
            for value in loader[filename]:
                batch_cnt[filename][value] = len(loader[filename][value])
        return batch_cnt

    def get_batch_idx(self, filename, value):
        if self.indices[filename][value] >= self.max_indices[filename][value]:
            self.loaders_ins[filename][value] = self.instantiate_one(self.loaders[filename][value])
            self.indices[filename][value] = 0

        return self.indices[filename][value]

    def get_filename(self):
        self.index = (self.index + 1) % len(self)
        return self.filenames[self.index]

    def combine_batch(self, neg_data, neg_attmask, neg_segid, neg_target,
                            pos_data, pos_attmask, pos_segid, pos_target):
        neg_data, pos_data = padding(neg_data, pos_data, pad_idx=self.pad_idx)
        neg_attmask, pos_attmask = padding(neg_attmask, pos_attmask, pad_idx=self.pad_idx)
        neg_segid, pos_segid = padding(neg_segid, pos_segid, pad_idx=self.pad_idx)
        # combine support data and query data
        
        # Batch size = 32, n_of_support = 5, n_of_query = 27
        # If you want to change the number of query then, change the batch size
        support_data = torch.cat([neg_data[0:self.support], pos_data[0:self.support]], dim=0)
        query_data = torch.cat([neg_data[self.support:], pos_data[self.support:]], dim=0)
        data = torch.cat([support_data, query_data], dim=0)

        # combine support mask and query mask
        support_attmask = torch.cat([neg_attmask[0:self.support], pos_attmask[0:self.support]], dim=0)
        query_attmask = torch.cat([pos_attmask[self.support:], pos_attmask[self.support:]], dim=0)
        attmask = torch.cat([support_attmask, query_attmask], dim=0)

        # combine support segid and query segid
        support_segid = torch.cat([neg_segid[0:self.support], pos_segid[0:self.support]], dim=0)
        query_segid = torch.cat([neg_segid[self.support:], pos_segid[self.support:]], dim=0)
        segid = torch.cat([support_segid, query_segid], dim=0)

        # combine support target and query target
        support_target = torch.cat([neg_target[0:self.support], pos_target[0:self.support]], dim=0)
        query_target = torch.cat([neg_target[self.support:], pos_target[self.support:]], dim=0)
        target = torch.cat([support_target, query_target], dim=0)
        return data, attmask, segid, target

    def get_batch(self):
        filename = self.get_filename()
        neg_idx = self.get_batch_idx(filename, 'neg')
        pos_idx = self.get_batch_idx(filename, 'pos')
        neg_data, neg_attmask, neg_segid, neg_target = self.loaders_ins[filename]['neg'][neg_idx]
        pos_data, pos_attmask, pos_segid, pos_target = self.loaders_ins[filename]['pos'][pos_idx]
        self.indices[filename]['neg'] += 1
        self.indices[filename]['pos'] += 1
        # imcomplete batch
        if min(len(neg_data), len(pos_data)) < self.support + self.query:
            return self.get_batch()
        data, attmask, segid, target = self.combine_batch(neg_data, neg_attmask, neg_segid, neg_target,
                                          pos_data, pos_attmask, pos_segid, pos_target)
        return data, attmask, segid, target
