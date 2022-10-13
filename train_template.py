
import os, sys, json, shutil

import numpy as np
import pandas as pd
from pandas._config import config
import torch
from torch import optim
from torch.nn import functional

from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter


from utils.config import Config
from utils.utils import prepare_logger, set_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Train_template():
    """Base Training template class.
    This is the template class for training. Any model variant trainers can inherit from this to reuse the 
    basic common functionality.
    Args:
        jsonpath (string):
            path to the JSON config file.
        log_file (string):
            name for the logging file
    """

    def __init__(self, jsonpath = None, log_file=None):    
        # Load the config from the config file
        config_obj = Config(jsonpath)
        config_obj.load_config((config_obj.__dict__))
        self.config = config_obj.input_argument
        
        # Common Model Params
        self._batch_size = self.config['commonModelParameter']['batch_size']
        assert self._batch_size % 2 == 0
        set_seed(self.config['commonModelParameter']['seed'])

        self._device = device
        self.logger = self.setup_logger(log_file) # set-up logger
    


    def setup_logger(self, log_file=None):
        # Prepare the logger path before creating the logger obj
        name = log_file if log_file != None else "log_"
        self.config['log_filename'] = os.path.join(self.config["loggingPaths"]['log_path'], name)
        if not os.path.exists(self.config["loggingPaths"]["log_path"]):
            os.makedirs(self.config['loggingPaths']["log_path"])
        logger = prepare_logger(self.config)
        return logger
        

    def print_train_args(self):
        # Print the training args
        self.logger.info(100 * '=')
        self.logger.info('> Training arguments:\n')
        self.logger.info((json.dumps(self.config, indent=2)))
        self.logger.info(100 * '=')


    def check_path_args(self):
        self.config["loggingPaths"]['vis_path'] = os.path.join(self.config["loggingPaths"]['vis_path'], self.config["loggingPaths"]['save_suffix'])
        self.config["loggingPaths"]['samples_save_path'] = os.path.join(self.config["loggingPaths"]['samples_save_path'], self.config["loggingPaths"]['save_suffix'])
        # Check all provided paths:
        if not os.path.exists(self.config["loggingPaths"]['model_save_path']):
            self.logger.info("Creating checkpoint path for saved models at:  {}\n".format(self.config["loggingPaths"]['model_save_path']))
            os.makedirs(self.config["loggingPaths"]['model_save_path'])
        else:
            self.logger.info("Model save path checked..")
        if not os.path.exists(self.config["loggingPaths"]['samples_save_path']):
            self.logger.info("Creating path for saving the gen samples at:  {}\n".format(self.config["loggingPaths"]['samples_save_path']))
            os.makedirs(self.config["loggingPaths"]['samples_save_path'])
        else:
            self.logger.info("Samples save path checked..")
        if not os.path.exists(self.config["loggingPaths"]['vis_path']):
            self.logger.info("Creating checkpoint path for Tensorboard visualizations at:  {}\n".format(self.config["loggingPaths"]['vis_path']))
            os.makedirs(self.config["loggingPaths"]['vis_path'])
        else:
            self.logger.info("Tensorboard Visualization path checked..")
            self.logger.info("Cleaning Visualization path of older tensorboard files...\n")
            for filename in os.listdir(self.config["loggingPaths"]['vis_path']):
                file_path = os.path.join(self.config["loggingPaths"]['vis_path'], filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    self.logger.info('Failed to delete %s. Reason: %s' % (file_path, e))
            # shutil.rmtree(self.config['vis_path'])
        self.writer = SummaryWriter(self.config["loggingPaths"]['vis_path'])


    def prepare_data_and_data_sampler(self):
        raise NotImplementedError
    

    def prepare_models(self):
        raise NotImplementedError
    

    def prepare_optimizer(self, seq=False):
        # Initialize Optimizers
        raise NotImplementedError
    

    def _run(self):
        """
        Each model will have it's own custom training function. 
        So we leave this blank here to be overwritten in the child class
        """
        raise NotImplementedError    
    

    def save_model(self):
        model_save_dict = {'config': self.config, 'model_state_dict': self.model.state_dict(), 'optim_state_dicts': [self.e_opt.state_dict(), self.r_opt, self.s_opt, self.g_opt, self.d_opt]}
        model_save_file = os.path.join(self.config['loggingPaths']['model_save_path'], self.config['loggingPaths']['save_suffix']+'_'+self.config['loggingPaths']['model_save_name'])
        self.logger.info("\nSaving Model at :  {}".format(model_save_file))
        torch.save(model_save_dict, model_save_file)
    


    def sample(self,n):
        """
        Sample data similar to the training data.
        """
        raise NotImplementedError
    



    

