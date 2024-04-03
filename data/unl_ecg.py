import os
import h5py
import hdf5plugin
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.fft as fft

from data.utils import load_ecg, ALL_LEADS


class UNLECGDataset(Dataset):
    def __init__(self, args, pid):
        self.args = args
        self.pid = pid
        self.nsamples = 2500
        self.n_channels = 12
        self.nviews = self.args.nviews

    def __len__(self):
        return len(self.pid)
    
    def obtain_perturbed_frame(self, frame):
        """ Apply Sequence of Perturbations to Frame 
        Args:
            frame (numpy array): frame containing ECG data (1250, 12)
        Outputs
            frame (numpy array): perturbed frame based
        """
        if self.args.gaussian:
            mult_factor = 1
            variance_factor = 0.1
            gauss_noise = np.random.normal(0,variance_factor,size=(frame.shape[0], frame.shape[1]))
            frame = frame + gauss_noise

        if self.args.flipalongx:
            frame = -frame

        if self.args.flipalongy:
            frame = np.flip(frame)
        return frame

    def normalize_frame(self,frame):
        if isinstance(frame,np.ndarray):
            frame = (frame - np.min(frame))/(np.max(frame) - np.min(frame) + 1e-8)
        elif isinstance(frame,torch.Tensor):
            frame = (frame - torch.min(frame))/(torch.max(frame) - torch.min(frame) + 1e-8)
        
        return frame
    
    def get_contrastive(self, x):
        '''
        Used for contrastive learning (SimCLR, CMSC)
        '''
       
        frame_views = []
        if self.args.contrastive_mode == 'cmsc':
            """ Start My Approach Patient Specific """
            # frame_views = torch.tensor(x,dtype=torch.float)
            
            start = 0
            for n in range(self.nviews):
                current_view = x[start:start+int(self.nsamples/self.nviews), :]
                current_view = self.obtain_perturbed_frame(current_view)
                current_view = self.normalize_frame(current_view)
                
                frame_views.append(current_view)
                
                start += 1250
            frame_views = np.transpose(np.array(frame_views), (2,1,0))

        elif self.args.contrastive_mode == 'simclr':
            
            frame_views = []
            for n in range(self.nviews):
                """ Obtain Differing 'Views' of Same Instance by Perturbing Input Frame """
                frame = self.obtain_perturbed_frame(x)    

                """ Normalize Data Frame """
                frame = self.normalize_frame(frame)
                frame_views.append(frame)

            frame_views = np.transpose(np.array(frame_views), (2,1,0))
        return frame_views

    def __getitem__(self, idx):
        ecg_pid = self.pid[idx]
        unl_id = ecg_pid.split('_')
        fname = os.path.join(self.args.dir_unl, unl_id[0], unl_id[1]+'.hd5')

        hd5 = h5py.File(fname, "r")
        x = load_ecg(hd5, unl_id[2]).astype(np.float32)
        x = x / 1000

        if self.args.contrastive:
            self.nsamples = x.shape[0]
            self.n_channels = x.shape[1]

            x = self.get_contrastive(x)
            return x, idx

        if self.args.dtw:
            return x.T, idx
        else:
            return x.T