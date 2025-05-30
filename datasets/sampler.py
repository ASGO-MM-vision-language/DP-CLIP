import os
import torch
import random
import math
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class CategorySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size,shuffle, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle  # True
        self.drop_last = drop_last  ##  True
        self.all_sketches_path = self.data_source.all_sketches_path

        self.category_sketch_index = defaultdict(list)

        for index, filepath in enumerate(self.all_sketches_path):
            category = filepath.split(os.path.sep)[-2]
            self.category_sketch_index[category].append(index)
        self.length = len(self._random_sampler())   

        
    def __iter__(self):
        indices = self._random_sampler()
        return iter(indices)
    

    def _random_sampler(self):
        indices = []
        for category in self.category_sketch_index.keys():
            index = torch.tensor(self.category_sketch_index[category])

            num_index = index.shape[0]
            rand_index = torch.randperm(num_index)
            index = index[rand_index]

            num_batch = int(num_index / self.batch_size)
            for i in range(num_batch):
                indices.append(index[i*self.batch_size : (i+1)*self.batch_size])
 
        ## random category index
        indices = torch.stack(indices, dim=0)
        rand_index = torch.randperm(indices.shape[0])
        rand_indices = indices[rand_index]

        indices = rand_indices.view(rand_indices.shape[0]*rand_indices.shape[1])

        return indices   
    

    def __len__(self):
        return self.length
  

class ValSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source):  # , batch_size,shuffle, drop_last
        self.data_source = data_source
        self.all_sketches_path = self.data_source.all_sketches_path

        self.category_sketch_index = defaultdict(list)
        self.category_index_list = []
        for index, filepath in enumerate(self.all_sketches_path):
            category = filepath.split(os.path.sep)[-2]
            self.category_sketch_index[category].append(index)
            self.category_index_list.append(category)

    def __iter__(self):
        indices = []
        for category in self.category_sketch_index.keys():
            index = torch.tensor(self.category_sketch_index[category])
            indices.append(index)
        
        print('val, len(indices):', len(indices))
        indices = torch.cat(indices, dim=0)
        print('val, len(indices):', len(indices))

        return iter(indices)

    def __len__(self):
        return len(self.data_source)


class ValBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = self.get_len()

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                # print('finish a batch when i =', i, 'idx = ', idx.item())
                yield batch
                batch = []
            
            if (i < len(sampler_list) - 1
                and self.sampler.category_index_list[idx] != self.sampler.category_index_list[sampler_list[i + 1]]
                ):
                if len(batch) > 0 and not self.drop_last:
                    # print('here', i, self.sampler.category_index_list[idx], self.sampler.category_index_list[sampler_list[i + 1]])
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1

        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def get_len(self):
        num_batch = 0

        for category in self.sampler.category_sketch_index.keys():
            index = torch.tensor(self.sampler.category_sketch_index[category])
            if self.drop_last:
                num_batch += int(len(index) / self.batch_size)
            else:
                num_batch += math.ceil(len(index) / self.batch_size)
            # print(category, len(index), num_batch)
        return num_batch


    def __len__(self):
        return self.length 
        