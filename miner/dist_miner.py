import os
import random
import numpy as np
import torch


class BatchMiner():
    def __init__(self, args):
        self.args          = args
        self.name          = 'dist_miner'

    def __call__(self, batch, dist_matrix, step):

        '''
        batch: ECG input as a batch
        dist_matrix: batch x batch 2D matrix, dtype - numpy
        '''

        if self.args.load_triplet:
            if self.args.dist_calc == 'euclidean':
                triplets = np.load(os.path.join(self.args.dir_euctriplets, str(step)+'.npy'))
            else:
                triplets = np.load(os.path.join(self.args.dir_triplets, str(step)+'.npy'))
            anchors = triplets[0]
            positives = triplets[1]
            
            if self.args.neg_random:
                negatives = []
                for i in range(len(batch)):
                    ancpos = []
                    ancpos.append(anchors[i])
                    ancpos.append(positives[i])
                    
                    negatives.append(np.random.choice(np.delete(range(len(batch)), ancpos)))
                    
            else:
                negatives = triplets[2]

        else:
            anchors, positives, negatives = [],[],[]
 
            for i in range(len(batch)):
                anchor   = i

                if self.args.matrix:
                    dist_copy = dist_matrix[i].clone()
                    dist_copy[i] = float('Inf')
                    positive = torch.argmin(dist_copy)
                
                else:
                    positive_indices = np.argpartition(dist_matrix[i], -2)[-2:]
                    positive = positive_indices[0]

                    if positive == i:
                        positive = positive_indices[1]

                if self.args.neg_random:
                    neg_idxs = [i for i in range(len(batch)) if (i!= positive and i!= anchor)]

                    negative = random.choice(neg_idxs)
                    while negative == i:
                        negative = random.choice(neg_idxs)
                    negatives.append(negative)
                
                else:
                    negative_indices = torch.topk(abs(dist_matrix[i]), self.args.neg_topk)[1]
                    negative = random.choice(negative_indices).item()

                    while (negative == anchor or negative == positive):
                        negative = random.choice(negative_indices).item()
                    negatives.append(negative)

                anchors.append(anchor)
                positives.append(positive.item())
            
        assert len(anchors) == len(positives) == len(negatives)
        
        if self.args.save_triplet:
            if self.args.dist_calc == 'euclidean':
                np.save(os.path.join(self.args.dir_euctriplets, str(step)+'.npy'), np.stack((anchors, positives, negatives)))
            elif self.args.dist_calc == 'dtwreal':
                np.save(os.path.join(self.args.dir_dtwrealtriplets, str(step)+'.npy'), np.stack((anchors, positives, negatives)))
            else:
                np.save(os.path.join(self.args.dir_triplets, str(step)+'.npy'), np.stack((anchors, positives, negatives)))

        sampled_triplets = [[a, p, n] for a, p, n in zip(anchors, positives, negatives)]
        
        return sampled_triplets

    def find_tripelts(self, args, batch, dist_matrix):

        return anchors, positives, negatives