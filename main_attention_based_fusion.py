"""
CIN classification - Part II
Attention based fusion - Image-level classifier
@author: Sudhir Sornapudi
@email: ssbw5@mst.edu

@date: March 4, 2020

NOTE:   Model might sometimes suffer from exploding gradients. 
        Solution: Try changing learning rate (and/or) use gradient cliping.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
import sys
from utils.utilities import matrix_mul, element_wise_mul
from utils.pytorchtools import EarlyStopping
import numpy as np
import time
import copy
import math
import os
import glob
import logging
from utils.utilities import initialize_logger
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef,average_precision_score,precision_score,recall_score,f1_score,balanced_accuracy_score,cohen_kappa_score
from torch.utils.tensorboard import SummaryWriter 

import models.att_model as att_model
import models.att_ops as ops

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

log_DIR = r'/usr/local/home/ssbw5/classification/deep-cin_expts/crnn/logs/new_im/mm_K5_models.crnn_pretrained_densenet121/logit_vec'
Kfolds = 5
classes = ['Normal', 'CIN1', 'CIN2', 'CIN3']

with_cuda = True
perform_train = False
perform_val = True

if perform_val:
    initialize_logger(os.path.join(log_DIR,'kappa.log'))
    logging.info("Attention based overall image classification")
    overall_im_class_acc = []
    overall_im_class_auc = []
    overall_im_class_mcc = []
    overall_im_class_PR = []
    overall_im_class_REC = []
    overall_im_class_F1 = []
    overall_im_class_AP = []
    overall_im_class_balanced_acc = []
    overall_im_class_k = []

for model_num in range(Kfolds):
    model = att_model.rnn_model(nclass=4, nh_im=128, with_cuda=with_cuda)

    if perform_train:
        print("Model {}, [{}/{}]".format(model_num,model_num+1,Kfolds))
        data_tr = torch.load(os.path.join(log_DIR,'train_feats_m'+str(model_num)+'.pt'))
        data_val = torch.load(os.path.join(log_DIR,'test_feats_m'+str(model_num)+'.pt'))

        data = {'train':(data_tr['x'],data_tr['y'],data_tr['patch_count'],data_tr['im_names']),
                'val':(data_val['x'],data_val['y'], data_val['patch_count'],data_val['im_names'])}
        # data = {'train':(data_tr['x'],data_tr['y'],data_tr['patch_count'],None),
        #         'val':(data_val['x'],data_val['y'], data_val['patch_count'],None)}

        class_sample_count = torch.tensor([(data_tr['y'] == t).sum() for t in torch.unique(data_tr['y'])])
        weight = 1. / class_sample_count.float()

        if torch.cuda.is_available() and with_cuda:
            model = model.to('cuda')
            weight = weight.to('cuda')
        criterion = nn.CrossEntropyLoss(weight)
        # try changng lr to 0.001, if loss is NaN
        optimizer = optim.SGD(model.parameters(), lr=0.0001)

        model,_,_,_ = ops.train_model(model_num, model, data, criterion, optimizer, hist_dir=log_DIR, 
                                    early_stop_patience=20, num_epochs=200, with_cuda=with_cuda)

    if perform_val:
        logging.info("Validating Model {}, [{}/{}]".format(model_num,model_num+1,Kfolds))

        data_val = torch.load(os.path.join(log_DIR,'test_feats_m'+str(model_num)+'.pt'))
        data = {'val':(data_val['x'],data_val['y'], data_val['patch_count'], data_val['im_names'])}
        # data = {'val':(data_val['x'],data_val['y'], data_val['patch_count'], None)}

        class_sample_count = torch.tensor([(data_val['y'] == t).sum() for t in torch.unique(data_val['y'])])
        weight = 1. / class_sample_count.float()

        model_wt_dir = glob.glob(os.path.join(log_DIR,'att_best','model_{}'.format(model_num),'*.pth'))
        print(model_wt_dir)
        if torch.cuda.is_available() and with_cuda:
            model.load_state_dict(torch.load(model_wt_dir[0]))
        else:    
            model.load_state_dict(torch.load(model_wt_dir[0],map_location=torch.device('cpu')))
    val_start_time = time.time()
    overall_im_labels, overall_im_probs, overall_gt,all_f,all_l = ops.infer(model, data['val'], with_cuda=with_cuda,verbose=False)
    overall_im_probs = overall_im_probs.reshape(-1,4)
    print("Validation_run_time: {} sec".format(time.time()-val_start_time))

    cm_4class = confusion_matrix(overall_gt,overall_im_labels)
    cm_2class = confusion_matrix(np.clip(overall_gt,0,1),np.clip(overall_im_labels,0,1))
    logging.info("classification CM - 4 class: \n{}".format(cm_4class))
    logging.info("classification CM - 2 class: \n{}".format(cm_2class))
    logging.info("\n"+classification_report(overall_gt, overall_im_labels, target_names=classes, digits=4))

    # Only for binarization of gt and preds
    if False:
        overall_gt = np.clip(overall_gt,0,1)
        overall_im_labels = np.clip(overall_im_labels,0,1)
        overall_im_probs = np.concatenate((overall_im_probs[:,0].reshape(-1,1),np.sum(overall_im_probs[:,1:],axis=1).reshape(-1,1)),axis=1)
        class_sample_count = torch.tensor([(torch.tensor(overall_gt) == t).sum() for t in torch.unique(torch.tensor(overall_gt))])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in overall_gt])
        auc = roc_auc_score(overall_gt, np.argmax(overall_im_probs,axis=1).reshape(-1,))

    #np.savez('gt_probs_{}.npz'.format(model_num),gt=overall_gt,probs=overall_im_probs)
    auc = roc_auc_score(overall_gt, overall_im_probs, average='weighted',multi_class='ovr')
    mcc = matthews_corrcoef(overall_gt, overall_im_labels)
    kappa = cohen_kappa_score(overall_gt, overall_im_labels)
    one_hot = np.zeros((overall_gt.size, overall_gt.max()+1))
    one_hot[np.arange(overall_gt.size),overall_gt] = 1
    ap = average_precision_score(one_hot, overall_im_probs, average='weighted')
    pr = precision_score(overall_gt, overall_im_labels, average='weighted')
    rec = recall_score(overall_gt, overall_im_labels, average='weighted')
    f1 = f1_score(overall_gt, overall_im_labels, average='weighted')
    acc = np.sum(overall_gt==overall_im_labels)/len(overall_gt)

    overall_im_class_acc.append(acc)
    overall_im_class_auc.append(auc)
    overall_im_class_mcc.append(mcc)
    overall_im_class_AP.append(ap)
    overall_im_class_PR.append(pr)
    overall_im_class_REC.append(rec)
    overall_im_class_F1.append(f1)
    overall_im_class_k.append(kappa)

    logging.info("classification acc: {:.2f}%".format(acc*100))
    #logging.info("AUC: {:.2f}%".format(auc*100))
    #logging.info("MCC: {:.2f}%".format(mcc*100))
    if perform_val:
        # train images on tensorboard 
        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(os.path.join(log_DIR,'runs/model_{}_input_data'.format(model_num)))
        # log embeddings
        print("embeddings:",all_f.size(),all_l.size())
        writer.add_embedding(all_f,
                            metadata=all_l,
                            global_step=model_num)
        del all_f
        del all_l
        writer.close()

logging.info("")
overall_im_class_acc = np.array(overall_im_class_acc)
overall_im_class_auc = np.array(overall_im_class_auc)
#overall_im_class_AP = np.array(overall_im_class_AP)
overall_im_class_PR = np.array(overall_im_class_PR)
overall_im_class_REC = np.array(overall_im_class_REC)
overall_im_class_F1 = np.array(overall_im_class_F1)
overall_im_class_mcc = np.array(overall_im_class_mcc)
overall_im_class_k = np.array(overall_im_class_k)

logging.info("Overall model performance")
logging.info("*"*25)
logging.info("Overall median acc: {:.2f}%".format(np.median(overall_im_class_acc)*100))
logging.info("Overall mean acc: {:.2f}%".format(np.mean(overall_im_class_acc)*100))
logging.info("Overall std acc: {:.2f}%".format(np.std(overall_im_class_acc)*100))
logging.info(" ")
logging.info("Overall mean AP: {:.2f}%".format(np.mean(overall_im_class_AP)*100))
logging.info("Overall std AP: {:.2f}%".format(np.std(overall_im_class_AP)*100))
logging.info(" ")
logging.info("Overall mean PRECISION: {:.2f}%".format(np.mean(overall_im_class_PR)*100))
logging.info("Overall std PRECISION: {:.2f}%".format(np.std(overall_im_class_PR)*100))
logging.info(" ")
logging.info("Overall mean RECALL: {:.2f}%".format(np.mean(overall_im_class_REC)*100))
logging.info("Overall std RECALL: {:.2f}%".format(np.std(overall_im_class_REC)*100))
logging.info(" ")
logging.info("Overall mean F1: {:.2f}%".format(np.mean(overall_im_class_F1)*100))
logging.info("Overall std F1: {:.2f}%".format(np.std(overall_im_class_F1)*100))
logging.info(" ")
logging.info("Overall mean ROC-AUC: {:.2f}%".format(np.mean(overall_im_class_auc)*100))
logging.info("Overall std ROC-AUC: {:.2f}%".format(np.std(overall_im_class_auc)*100))
logging.info(" ")
logging.info("Overall mean MCC: {:.2f}%".format(np.mean(overall_im_class_mcc)*100))
logging.info("Overall std MCC: {:.2f}%".format(np.std(overall_im_class_mcc)*100))
logging.info(" ")
logging.info("Overall mean kappa: {:.2f}%".format(np.mean(overall_im_class_k)*100))
logging.info("Overall std kappa: {:.2f}%".format(np.std(overall_im_class_k)*100))
