import numpy as np
import sampler.class_random_sampler

def select(sampler, args, **kwargs):
    train_ids = np.load("./stores/train_ids.npy")
    train_label = np.load("./stores/train_label.npy")
    
    if 'random' in sampler:
        if 'class' in sampler:
            sampler_lib = class_random_sampler
    else:
        raise Exception('Minibatch sampler <{}> not available!'.format(sampler))
    sampler = sampler_lib.Sampler(args,label_list=train_label, train_list=train_ids)
           
    return sampler