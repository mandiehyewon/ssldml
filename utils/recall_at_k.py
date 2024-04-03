'''
Code Retrieved From: 
https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/metrics/e_recall.py
Author: Karsten Roth
Revisiting Training Strategies and Generalization Performance in Deep Metric Learning
'''

## future work: need to implement c_recall_1 later on
import numpy as np
import faiss

import torch

class Recall_At_K():
    def __init__(self, k, **kwargs):
        self.k        = k
        self.requires = ['nearest_features', 'target_labels']
        self.name     = 'e_recall@{}'.format(k)

    def __call__(self, target_labels, features, test=False, *kwargs):
        k_closest_classes = self.calculate_features(target_labels, features)
        recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:self.k]])/len(target_labels)
        precision_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:self.k]])/len(target_labels)
        
        return recall_at_k

    def calculate_features(self, target_labels, features, **kwargs):
        '''
        Compute Nearest Neighbors
        '''
        faiss.omp_set_num_threads(6)
        # faiss.omp_set_num_threads(self.pars.kernels)
        torch.cuda.empty_cache()
        res = faiss.StandardGpuResources()
    
        faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
        if res is not None:
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
        faiss_search_index.add(features)
        _, k_closest_points = faiss_search_index.search(features, int(self.k+1))
        k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

        return k_closest_classes