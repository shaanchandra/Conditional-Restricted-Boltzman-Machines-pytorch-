import warnings
warnings.filterwarnings('ignore')

import argparse, time, datetime
import os, sys, json
import pickle

import numpy as np
import pandas as pd
import torch
from packaging import version
# from sklearn.preprocessing import MinMaxScaler
from torch import optim
from sklearn.model_selection import train_test_split


from train_template import Train_template
from data_utils.data_preprocess_crbm import data_preprocess
from data_utils.data_loader_crbm import CRBMDataset
from crbm import CRBM, Forest
from utils.utils import set_seed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CRBM_trainer(Train_template):
    """
    Conditional Restricted Boltzman Machines Training class.
    This is the core class of the CRBM training, which inherits its base properties from the template class,
    and the custom training modules are defined here.
    """

    def __init__(self, jsonpath, log_file=None, load_pretrained=False):
        super(CRBM_trainer, self).__init__(jsonpath, log_file, load_pretrained)

        # Common model param
        self.adv_wt = self.config['crbmParams']['adv_wt']
        self._verbose = self.config['commonModelParameter']['verbose']
        self.verbose_step = self.config['commonModelParameter']['verbose_step']
        self.lr = self.config['crbmParams']['lr']
        self.epochs = self.config['commonModelParameter']['epochs']

        self.print_train_args()
        self.check_path_args() # Check all the provided paths 
        self.prepare_data_and_data_sampler() # Check the data and create the data sampler
        self.prepare_models() # Initialize the G and D model architectures
        self.prepare_optimizer() # Initialize the optimizers for training
    

    def prepare_data_and_data_sampler(self):
        """
        Performs:
            1. Data preprocessing
            2. Sets model parameters based on the processed data (feat_dims, etc)
            3. Prepares data loaders
        """       
        # Train-Test Split data (for encoder, decoder, supervisor, generator, discriminator)
        self.logger.info("\n>> Data pre-processsing and data loader preparation....")
        vis_mtx, cond_mtx, self.config['scaler_seq'], self.config['scaler_static'], self.config['commonModelParameter']['max_seq_len'], self.config['commonModelParameter']['padding_value'] = data_preprocess(self.config, self.logger)
        self.config['crbmParams']['input_dim'] = vis_mtx.shape[1]
        self.config['crbmParams']['cond_dim'] = cond_mtx.shape[1]
        # self.train_data_vis, self.train_data_cond = self.custom_train_test_split(vis_mtx, cond_mtx)
        train_dataset = CRBMDataset(vis_mtx, cond_mtx)
        self.train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.config['commonModelParameter']['batch_size'], shuffle=False)
                
    

    def prepare_models(self):
        """
        Initialize the model based on config settings
        """
        self.logger.info("\n\n" + "="*75 + "\n\tCRBM Model details\n" + "="*75)
        self.crbm = CRBM(self.config)
        if self._verbose:
            self.logger.info("\nCRBM = {}\n".format(self.crbm))
        num_params_crbm = sum(p.numel() for p in self.crbm.parameters() if p.requires_grad)
        self.logger.info("-"*100 + "\nTotal no. of params =  {}".format(num_params_crbm))
    
    def prepare_optimizer(self, seq=False):
        # Initialize Optimizers
        self.opt_crbm = torch.optim.Adam(self.crbm.parameters(), lr=self.lr)
        


    def _run(self):
        """
        Main training module for CRBM training.
        """
        if self._verbose:
            self.logger.info("\n\n" + "="*100 + "\n\t\t\t\t\t Training Network\n" + "="*100)
            self.logger.info("\nBeginning training at:  {} \n".format(datetime.datetime.now()))
        
        for epoch in range(self.epochs):
            self.crbm.train()
            for v, cond in self.train_data_loader:
                self.crbm.zero_grad()
                v_gibb = self.crbm(v, cond)
                recon_loss = self.crbm.reconstruction_loss(v, v_gibb)
                recon_loss.backward()
                self.opt_crbm.step()
            if epoch%self.verbose_step==0:
                print("EPOCH= {},  LOSS = {:.3f}".format(epoch+1,recon_loss))

        # Save model
        self.save_model()
        self.logger.info("\nTraining and saving model finished !! Generating and saving synthetic seqs now....")

        # Load the saved model and run generation
        self.logger.info("\nLoading saved model for generation.....")
        model_save_dict = torch.load(os.path.join(self.config['loggingPaths']['model_save_path'], self.config['loggingPaths']['save_suffix']+'_'+self.config['loggingPaths']['model_save_name']))
        self.config = model_save_dict['config']
        self.model = CRBM(self.config)
        self.model.load_state_dict(model_save_dict['model_state_dict'])
        self.model.eval()        
        generated_data = self.sample(n=1000, save_df=True, rescale=True)



    def sample(self, n=None, save_df=True, rescale=False):
        raise NotImplementedError


    

if __name__=='__main__':

    crbm_trainer = CRBM_trainer(jsonpath = "./config_crbm.json")
    crbm_trainer._run()



