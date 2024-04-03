from itertools import combinations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = args.eps
        
    def forward(self,yhat,y):
        if yhat.size() == y.size():
            loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        else: # when y.size() == (batch, )
            loss = torch.sqrt(self.mse(yhat,y.unsqueeze(1)) + self.eps)
        return loss

def get_loss(args):
    if args.train_mode == 'regression':
        return RMSELoss(args)
        
    elif args.train_mode == 'binary_class':
        return nn.BCEWithLogitsLoss()

def obtain_contrastive_loss(args, latent_embeddings, pids):
    """ 
    Created on Sat May 16 23:28:57 2020
    @author: Dani Kiyasseh


    Used for SimCLR and CLOCS pretraining. 
    Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        latent_embeddings (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN)
        pids (list): patient ids of instances in batch
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    """
    if args.contrastive_mode in ['cmsc','CMLC','CMSMLC']:
        
        pids = [str(pid) for pid in list(pids)]
        pids = np.array(pids,dtype=np.object)
        pid1,pid2 = np.meshgrid(pids,pids)
        pid_matrix = pid1 + '-' + pid2
        pids_of_interest = np.unique(pids + '-' + pids) #unique combinations of pids of interest i.e. matching
        bool_matrix_of_interest = np.zeros((len(pids),len(pids)))
        
        for pid in pids_of_interest:
            bool_matrix_of_interest += pid_matrix == pid

        rows1,cols1 = np.where(np.triu(bool_matrix_of_interest,1))
        rows2,cols2 = np.where(np.tril(bool_matrix_of_interest,-1))

    nviews = set(range(latent_embeddings.shape[2]))
    view_combinations = combinations(nviews, 2)
    loss = 0
    ncombinations = 0
    
    for combination in view_combinations:
        view1_array = latent_embeddings[:,:,combination[0]] #(BxH)
        view2_array = latent_embeddings[:,:,combination[1]] #(BxH)
        
        norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
        norm2_vector = view2_array.norm(dim=1).unsqueeze(0)

        sim_matrix = torch.mm(view1_array,view2_array.transpose(0,1))
        norm_matrix = torch.mm(norm1_vector.transpose(0,1),norm2_vector)

        temperature = 0.1
        argument = sim_matrix/(norm_matrix*temperature)
        sim_matrix_exp = torch.exp(argument)
        
        if args.contrastive_mode == 'cmc':
            """ Obtain Off Diagonal Entries """

            #upper_triangle = torch.triu(sim_matrix_exp,1)
            #lower_triangle = torch.tril(sim_matrix_exp,-1)
            #off_diagonals = upper_triangle + lower_triangle

            diagonals = torch.diag(sim_matrix_exp)

            """ Obtain Loss Terms(s) """
            loss_term1 = -torch.mean(torch.log(diagonals/torch.sum(sim_matrix_exp,1)))
            loss_term2 = -torch.mean(torch.log(diagonals/torch.sum(sim_matrix_exp,0)))
            loss += loss_term1 + loss_term2 
            loss_terms = 2
            
        elif args.contrastive_mode == 'simclr':
            self_sim_matrix1 = torch.mm(view1_array,view1_array.transpose(0,1))
            self_norm_matrix1 = torch.mm(norm1_vector.transpose(0,1),norm1_vector)
            temperature = 0.1
            argument = self_sim_matrix1/(self_norm_matrix1*temperature)
            self_sim_matrix_exp1 = torch.exp(argument)
            self_sim_matrix_off_diagonals1 = torch.triu(self_sim_matrix_exp1,1) + torch.tril(self_sim_matrix_exp1,-1)
            
            self_sim_matrix2 = torch.mm(view2_array,view2_array.transpose(0,1))
            self_norm_matrix2 = torch.mm(norm2_vector.transpose(0,1),norm2_vector)
            temperature = 0.1
            argument = self_sim_matrix2/(self_norm_matrix2*temperature)
            self_sim_matrix_exp2 = torch.exp(argument)
            self_sim_matrix_off_diagonals2 = torch.triu(self_sim_matrix_exp2,1) + torch.tril(self_sim_matrix_exp2,-1)

            denominator_loss1 = torch.sum(sim_matrix_exp,1) + torch.sum(self_sim_matrix_off_diagonals1,1)
            denominator_loss2 = torch.sum(sim_matrix_exp,0) + torch.sum(self_sim_matrix_off_diagonals2,0)
            
            diagonals = torch.diag(sim_matrix_exp)
            loss_term1 = -torch.mean(torch.log(diagonals/denominator_loss1))
            loss_term2 = -torch.mean(torch.log(diagonals/denominator_loss2))
            loss += loss_term1 + loss_term2
            loss_terms = 2

        elif args.contrastive_mode in ['cmsc','cmlc','cmsmlc']: #ours #CMSMLC = positive examples are same instance and same patient
            triu_elements = sim_matrix_exp[rows1,cols1]
            tril_elements = sim_matrix_exp[rows2,cols2]
            diag_elements = torch.diag(sim_matrix_exp)
            
            triu_sum = torch.sum(sim_matrix_exp,1)
            tril_sum = torch.sum(sim_matrix_exp,0)
            
            loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
            loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))
            
            loss_triu = -torch.mean(torch.log(triu_elements/triu_sum[rows1]))
            loss_tril = -torch.mean(torch.log(tril_elements/tril_sum[cols2]))
            
            loss = loss_diag1 + loss_diag2
            loss_terms = 2

            if len(rows1) > 0:
                loss += loss_triu #technically need to add 1 more term for symmetry
                loss_terms += 1
            
            if len(rows2) > 0:
                loss += loss_tril #technically need to add 1 more term for symmetry
                loss_terms += 1
        
            #print(loss,loss_triu,loss_tril)

        ncombinations += 1
    loss = loss/(loss_terms*ncombinations)

    return loss