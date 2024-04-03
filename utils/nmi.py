'''
Code Retrieved From: 
https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/metrics/nmi.py
https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/metrics/__init__.py
Author: Karsten Roth
Revisiting Training Strategies and Generalization Performance in Deep Metric Learning
'''
import faiss
from sklearn import metrics

import torch


class NMI():
    def __init__(self, args, **kwargs):
        self.requires = ['kmeans_nearest', 'target_labels']
        self.name     = 'nmi'
        self.args = args

    def __call__(self, target_labels, features):
        computed_cluster_labels = self.computed_cluster_labels(features)
        NMI = metrics.cluster.normalized_mutual_info_score(computed_cluster_labels.reshape(-1), target_labels.reshape(-1))
        return NMI

    def computed_cluster_labels(self, features):
        '''
        compute cluster labels
        '''
        faiss.omp_set_num_threads(6)
        # faiss.omp_set_num_threads(self.pars.kernels)
        torch.cuda.empty_cache()
        res = faiss.StandardGpuResources()
        
        centroids = self.compute_kmeans(features, res)
        faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
        if res is not None: 
            faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)

        faiss_search_index.add(centroids)
        _, computed_cluster_labels = faiss_search_index.search(features, 1)

        return computed_cluster_labels
    
    def compute_kmeans(self, features, res=None):
        n_classes = self.args.num_classes

        cluster_idx = faiss.IndexFlatL2(features.shape[-1])
        if res is not None:
            cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
        kmeans            = faiss.Clustering(features.shape[-1], n_classes)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000
        ### Train Kmeans
        kmeans.train(features, cluster_idx)
        centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_classes, features.shape[-1])
        
        return centroids