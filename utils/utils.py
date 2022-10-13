import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random
import os
import time, datetime
from time import strftime, localtime

from utils.logger import create_logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import logging
import sys
import torch.nn as nn

# from metrics.nll import NLL


    
def prepare_logger(config):
    # ===log===
    log_time_str = strftime("%m%d_%H%M_%S", localtime())
    # log_filename = strftime("log/log_%s" % log_time_str)
    log_filename = config['log_filename']
    if os.path.exists(log_filename + '.txt'):
        i = 2
        while True:
            if not os.path.exists(log_filename + '_%d' % i + '.txt'):
                log_filename = log_filename + '_%d' % i
                break
            i += 1
    log_filename = log_filename + '.txt'
    log = create_logger(__name__, silent=False, to_disk=True, log_file=log_filename)
    return log



def set_seed(seed):
    # Seeds for reproduceable runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def convert_date(date):
    return time.mktime(datetime.datetime.strptime(date, "%m/%d/%Y %H:%M").timetuple())



def log_tensorboard(config, writer, model, epoch, total_iters, loss, metrics=None, lr=0, thresh=0, loss_only=False, val=False):           
    if loss_only:
        writer.add_scalar('Train/Loss', sum(loss)/len(loss), total_iters)
        # if iters%1 == 0:
        #     for name, param in model_log.encoder.named_parameters():
        #         print("\nparam {} grad = {}".format(name, param.grad.data.view(-1)))
        #         sys.exit()
        #         if not param.requires_grad or param.grad is None:
        #             continue
        #         writer.add_histogram('Iters/'+name, param.data.view(-1), global_step= ((iters+1)+total_iters))
        #         writer.add_histogram('Grads/'+ name, param.grad.data.view(-1), global_step = ((iters+1)+ total_iters))
    else:
        if not val:
            writer.add_scalar('Train/Epoch_Loss', sum(loss)/len(loss), total_iters)
            writer.add_scalar('Train/F1', metrics['F1'], epoch)
            writer.add_scalar('Train/AUCROC', metrics['aucroc'], epoch)
            writer.add_scalar('Train/PRAUC', metrics['prauc'], epoch)
            writer.add_scalar('Train/Precision', metrics['precision'], epoch)
            writer.add_scalar('Train/Recall', metrics['recall'], epoch)
            writer.add_scalar('Train/Accuracy', metrics['accuracy'], epoch)
            writer.add_scalar("Stats/learning_rate", lr, epoch)
            writer.add_scalar("Stats/Optimal_thresh", metrics['optimal_threshold'], epoch)
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                writer.add_histogram('Epochs/' + name, param.data.view(-1), global_step= epoch)
        
            
        else:
            writer.add_scalar('Validation/Loss', loss, epoch)
            writer.add_scalar('Validation/F1', metrics['F1'], epoch)
            writer.add_scalar('Validation/AUCROC', metrics['aucroc'], epoch)
            writer.add_scalar('Validation/PRAUC', metrics['prauc'], epoch)
            writer.add_scalar('Validation/Recall', metrics['recall'], epoch)
            writer.add_scalar('Validation/Precision', metrics['precision'], epoch)
            writer.add_scalar('Validation/Accuracy', metrics['accuracy'], epoch)


