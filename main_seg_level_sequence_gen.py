""" 
CIN classification - Part I
Segment-level sequence generator
@author Sudhir Sornapudi
@email: ssbw5@mst.edu

@date: March 4, 2020
"""

from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter 
import logging
#from warpctc_pytorch import CTCLoss
import os
import glob
import time
import sys
import copy
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report,matthews_corrcoef,precision_score,recall_score,f1_score
import pandas as pd

import models.crnn_pretrained as crnn
from utils.pytorchtools import EarlyStopping
from utils.utilities import initialize_logger
from utils.utilities import CustomDataset
from utils.visualize import *
from models.ops import *

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

print(crnn.__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default=r'/usr/local/home/ssbw5/classification/deep-cin_expts/data/sz64Segs_new', 
                    help='path to train and validation data')
parser.add_argument('--resize', action='store_false', help='Whether to resize image')
parser.add_argument('--log_dir', default=r'/usr/local/home/ssbw5/classification/crnn/logs/new_im/mm_att', 
                    help='path to logs')
parser.add_argument('--validate', action='store_false', help='Run ONLY VALIDATION process')
parser.add_argument('--val_log_dir', default=r'/usr/local/home/ssbw5/classification/deep-cin_expts/crnn/logs/new_im/mm_K5_models.crnn_pretrained_densenet121', 
                    help='path to load previous logs [ONLY VALIDATION]')
parser.add_argument('--model_name', default='densenet121', help="name of model [ONLY for PRETRAINED models]")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=56, help='input batch size')
parser.add_argument('--blstm', action='store_false', help='Whether to use BLSTM')
parser.add_argument('--attention', action='store_true', help='Whether to use BLSTM + attention')
parser.add_argument('--imgH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=704, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train')
parser.add_argument('--kFold_splits', type=int, default=5, help='Num of splits for K-fold cross-validation')
parser.add_argument('--early_stop_patience', type=int, default=10, help='patience for early stopping')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--crnn', default='', help="path to crnn (to resume training)")
parser.add_argument('--save_feats', action='store_true', help='Save only logit vectors - Runs on validation')
parser.add_argument('--with_cuda', action='store_false', help='Whether to resize image')
opt = parser.parse_args()

save_feats = opt.save_feats

opt.log_dir = opt.log_dir+'_K'+str(opt.kFold_splits)+'_'+str(crnn.__name__)+'_'+str(opt.model_name)
if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)

if not save_feats:
    if not opt.validate:
        initialize_logger(os.path.join(opt.log_dir,'LogonTrain.log'))
    else:
        initialize_logger(os.path.join(opt.val_log_dir,'LogonONLYValidation.log'))
else:
    opt.validate = True
    print("Saving LOGITS")
    initialize_logger(os.path.join(opt.log_dir,'temp.log'))

logging.info("{}".format(opt))

classes = ['Normal', 'CIN1', 'CIN2', 'CIN3']

# Define Seeds
manualSeed = 0 #7439 #random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
#cudnn.deterministic = True
cudnn.benchmark = True
cudnn.enabled = True
def _init_fn():
    np.random.seed(manualSeed)

# Enable GPU if with_cuda flag is activated
if torch.cuda.is_available() and opt.with_cuda:
    print("WARNING: You have a CUDA device, so running with --cuda")
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

# Get Image data to perform image based split for KFold cross-validation 
# Create training and validation datasets
image_DIR = os.listdir(opt.train_dir)

gt_csv_DIR = r"/usr/local/home/ssbw5/classification/deep-cin_expts/data/ImagesAllImgLabelDrZuna.csv"
df_labels = pd.read_csv(gt_csv_DIR)
im_csv_list = df_labels.loc[:,"imName"].values
gt_csv_list = df_labels.loc[:,"imClass"].values-1 # -1 to make the class label start from 0

data = image_DIR
target = []
for im_name in data:
    idx = np.where(im_csv_list == im_name)
    if idx[0].shape[0]: # if there is a match shape[0] should 1, if not 0
        target.append(gt_csv_list[idx[0][0]])
    else:
        target.append(None)

data = np.array(data)
target = np.array(target)

nclass = len(np.unique(target))
nc = 3 # Num of input channels

image = torch.FloatTensor(opt.batchSize, nc, opt.imgH, opt.imgH)

if torch.cuda.is_available() and opt.with_cuda:
    image = image.to('cuda')
    
image = Variable(image)

best_validation_accuracy = []
best_val_error = []
overall_im_class_acc = []
model_num = 0
test_preds = np.array([])
test_gt = np.array([])

overall_im_class_mcc = []
overall_im_class_PR = []
overall_im_class_REC = []
overall_im_class_F1 = []

# K-fold Cross validation 
skf = StratifiedKFold(n_splits=opt.kFold_splits, shuffle=True, random_state=manualSeed)
for i, (train_index, test_index)in enumerate(skf.split(data,target)):
    logging.info("")
    #logging.info("*"*25)
    logging.info("Model {}, [{}/{}]".format(model_num,model_num+1,opt.kFold_splits))
    logging.info("*"*25)
    logging.info("Image level split...")
    logging.info("TRAIN: {}, TEST: {}".format(train_index.shape, test_index.shape))

#%%        
    # refresh and train a complete new model for each fold
    import models.crnn_pretrained as crnn
    crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh, opt.batchSize, model_name=opt.model_name, with_BLSTM=opt.blstm, with_attention=opt.attention,cuda=opt.with_cuda)
    if torch.cuda.is_available() and opt.with_cuda:
        crnn = crnn.to('cuda')
    # setup optimizer
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
#%%    
    if not opt.validate:
        # Compute class distribution at image level
        logging.info("Class distribution at image level...")
        class_sample_count = torch.tensor(
            [(target[train_index] == t).sum() for t in np.unique(target)])
        logging.info("train_class {}: {}".format(np.unique(target),class_sample_count))

        class_sample_count_test = torch.tensor(
            [(target[test_index] == t).sum() for t in np.unique(target)])
        logging.info("test_class {}: {}".format(np.unique(target),class_sample_count_test))

        # Extract patch info from image data
        tr_patch_data = [os.path.join(opt.train_dir,d,im_patch_name) for d in data[train_index] 
                            for im_patch_name in os.listdir(os.path.join(opt.train_dir,d)) if im_patch_name.endswith('tif')]
        tr_patch_target = [target[train_index][i] for i, d in enumerate(data[train_index]) 
                            for im_patch_name in os.listdir(os.path.join(opt.train_dir,d)) if im_patch_name.endswith('tif')]
        
        vl_path_data = [os.path.join(opt.train_dir,d,im_patch_name) for d in data[test_index] 
                            for im_patch_name in os.listdir(os.path.join(opt.train_dir,d)) if im_patch_name.endswith('tif')]
        vl_patch_target = [target[test_index][i] for i, d in enumerate(data[test_index]) 
                            for im_patch_name in os.listdir(os.path.join(opt.train_dir,d)) if im_patch_name.endswith('tif')] 
        
        # Compute class distribution at patch level
        logging.info("Class distribution at patch level...")
        class_sample_count = torch.tensor(
            [(tr_patch_target == t).sum() for t in np.unique(target)])
        logging.info("train_class {}: {}".format(np.unique(target),class_sample_count))

        class_sample_count_test = torch.tensor(
            [(vl_patch_target == t).sum() for t in np.unique(target)])
        logging.info("test_class {}: {}".format(np.unique(target),class_sample_count_test))

        # Compute samples weight (each patch should get its own weight) -- only on train data
        weight = 1. / class_sample_count.float()
        #weight = (1. - class_sample_count.float()/torch.sum(class_sample_count)) + 1.
        print("weight: ", weight)
        samples_weight = torch.tensor([weight[t] for t in tr_patch_target])
        print("samples_weight: ",samples_weight)

        # Create sampler, dataset, loader
        # produce equal class distribution equaling to maximum occurances of any class
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight,len(samples_weight))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, 
                                        torch.max(class_sample_count).item()*len(classes))   
        
        subset_train_dataset = CustomDataset(tr_patch_data, tr_patch_target, phase = 'train', resize=opt.resize)
        subset_test_dataset = CustomDataset(vl_path_data, vl_patch_target, phase = 'val', resize=opt.resize)
        subset_datasets = {'train': subset_train_dataset,
                            'val': subset_test_dataset}
        dataloaders_dict = {x: torch.utils.data.DataLoader(subset_datasets[x], batch_size=opt.batchSize, 
                        shuffle=False, sampler = sampler if x=='train' else None, num_workers=opt.workers) for x in ['train', 'val']}
    #%% Iterate DataLoader and check class balance for each batch
        logging.info("Class distribution at patch level after sampling...")
        im_running_count = torch.tensor([0, 0, 0, 0])
        for i, (x, y) in enumerate(dataloaders_dict['train']):
            im_running_count += torch.tensor([(y == 0).sum(), (y == 1).sum(), (y == 2).sum(), (y == 3).sum()])
            # print("batch index {}, 0/1/2/3: {}/{}/{}/{}".format(i, (y == 0).sum(), (y == 1).sum(), (y == 2).sum(), (y == 3).sum()))
        logging.info("train_class {}: {}".format(np.unique(target),im_running_count))
        num_tr_samples = im_running_count.sum().item()
        #print("im_running_count with sampler: ", im_running_count)
        #print("total imgs from batches with sampler: ", im_running_count.sum())
        #print("actual class distribution: ", class_sample_count)
        #print("actual #patches: ", len(tr_patch_data))
        print()
#%%
        # train images on tensorboard 
        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(os.path.join(opt.log_dir,'runs/model_{}_input_data'.format(model_num)))
        # get some random training images
        dataiter = iter(dataloaders_dict['train'])
        images, labels = dataiter.next()
        if torch.cuda.is_available() and opt.with_cuda:
            images = images.to('cuda')
            labels = labels.to('cuda')
        # create grid of images
        img_grid = utils.make_grid(images)
        # show images
        matplotlib_imshow(img_grid)
        # write to tensorboard
        writer.add_image('CIN_images', img_grid)
        # visualize the model
        #writer.add_graph(crnn, images)
    
#%% Train and test models
        #crnn.apply(weights_init)
        #criterion = nn.LogSoftmax(dim=0)
        criterion = nn.CrossEntropyLoss()
        #weight = class_sample_count.float()/torch.sum(class_sample_count)
        #criterion = FocalLoss(class_num=len(classes),alpha=weight,gamma=2)
        if torch.cuda.is_available() and opt.with_cuda:
            criterion = criterion.to('cuda')

    # Train the model
        best_model,best_val_acc,best_val_er,_,all_f,all_l = train_model(
            model_num, crnn, dataloaders_dict, num_tr_samples, criterion, optimizer, 
            opt.log_dir, opt.early_stop_patience, writer, num_epochs=opt.niter, with_cuda=opt.with_cuda)
        del dataloaders_dict
        best_validation_accuracy.append(best_val_acc)
        best_val_error.append(best_val_er)

        # log embeddings
        print("embeddings:",all_f.size(),all_l.size())
        writer.add_embedding(all_f,
                            metadata=all_l,
                            global_step=model_num)
        del all_f
        del all_l
        writer.close()

        nn_data_dir = os.path.join(opt.log_dir,'logit_vec')
        if not os.path.exists(nn_data_dir):
            os.mkdir(nn_data_dir)

        train_x_feats, tr_patch_count = infer_feats(model_num, best_model, data[train_index], target[train_index],
                         opt.train_dir, classes,resize =opt.resize, cuda=opt.with_cuda)
        train_feats = {'x':train_x_feats,'y':torch.tensor(target[train_index]),'patch_count':tr_patch_count,'im_names':data[train_index]} 
        torch.save(train_feats,os.path.join(nn_data_dir,'train_feats_m'+str(model_num)+'.pt'))
        test_x_feats, te_patch_count = infer_feats(model_num, best_model, data[test_index], target[test_index],
                         opt.train_dir, classes,resize =opt.resize,cuda=opt.with_cuda)     
        test_feats = {'x':test_x_feats,'y':torch.tensor(target[test_index]),'patch_count':te_patch_count,'im_names':data[test_index]}  
        torch.save(test_feats,os.path.join(nn_data_dir,'test_feats_m'+str(model_num)+'.pt'))  

    else:
        best_model = crnn
        # original saved file with DataParallel
        model_wt_dir = glob.glob(os.path.join(opt.val_log_dir,'model_{}'.format(model_num),'*.pth'))
        if not opt.with_cuda:
            best_model.load_state_dict(torch.load(model_wt_dir[0], map_location=lambda storage, loc: storage))
        else:
            best_model.load_state_dict(torch.load(model_wt_dir[0]))

    if save_feats:
        nn_data_dir = os.path.join(opt.val_log_dir,'logit_vec')
        if not os.path.exists(nn_data_dir):
            os.mkdir(nn_data_dir)

        train_x_feats, tr_patch_count = infer_feats(model_num, best_model, data[train_index], target[train_index],
                         opt.train_dir, classes,resize =opt.resize, cuda=opt.with_cuda)
        train_feats = {'x':train_x_feats,'y':torch.tensor(target[train_index]),'patch_count':tr_patch_count,'im_names':data[train_index]} 
        torch.save(train_feats,os.path.join(nn_data_dir,'train_feats_m'+str(model_num)+'.pt'))
        test_x_feats, te_patch_count = infer_feats(model_num, best_model, data[test_index], target[test_index],
                         opt.train_dir, classes,resize =opt.resize,cuda=opt.with_cuda)     
        test_feats = {'x':test_x_feats,'y':torch.tensor(target[test_index]),'patch_count':te_patch_count,'im_names':data[test_index]}  
        torch.save(test_feats,os.path.join(nn_data_dir,'test_feats_m'+str(model_num)+'.pt'))            

    else:
        # Validate model for PR-curves
        data_val = torch.load(os.path.join(opt.val_log_dir,'logit_vec','test_feats_m'+str(model_num)+'.pt'))
        data = {'val':(data_val['x'],data_val['y'], data_val['patch_count'], data_val['im_names'])}
        val_start_time = time.time()
        overall_im_labels, overall_gt = infer_avg_vote(model_num, best_model, data_val['im_names'], data_val['y'],
                            opt.train_dir, classes,resize=opt.resize,cuda=opt.with_cuda)
        logging.info("Validation_run_time: {} sec".format(time.time()-val_start_time))
        del best_model
        del crnn

        test_preds = np.concatenate((test_preds,overall_im_labels),axis=0)
        test_gt = np.concatenate((test_gt,overall_gt),axis=0)
        acc = np.sum(overall_gt==overall_im_labels)/len(overall_gt)
        cm_4class = confusion_matrix(overall_gt,overall_im_labels)
        cm_2class = confusion_matrix(np.clip(overall_gt,0,1),np.clip(overall_im_labels,0,1))

        mcc = matthews_corrcoef(overall_gt, overall_im_labels)
        pr = precision_score(overall_gt, overall_im_labels, average='weighted')
        rec = recall_score(overall_gt, overall_im_labels, average='weighted')
        f1 = f1_score(overall_gt, overall_im_labels, average='weighted')

        overall_im_class_mcc.append(mcc)
        overall_im_class_PR.append(pr)
        overall_im_class_REC.append(rec)
        overall_im_class_F1.append(f1)

        overall_im_class_acc.append(acc)
        logging.info("classification CM - 4 class: \n{}".format(cm_4class))
        logging.info("classification CM - 2 class: \n{}".format(cm_2class))
        logging.info("\n"+classification_report(overall_gt, overall_im_labels, target_names=classes))
        logging.info("classification acc: {:.2f}%".format(acc*100))
    model_num = model_num + 1

if not save_feats:
    # Print metrics
    logging.info("")
    overall_im_class_acc = np.array(overall_im_class_acc)
    overall_im_class_PR = np.array(overall_im_class_PR)
    overall_im_class_REC = np.array(overall_im_class_REC)
    overall_im_class_F1 = np.array(overall_im_class_F1)
    overall_im_class_mcc = np.array(overall_im_class_mcc)

    logging.info("Image-level Overall model performance")
    logging.info("*"*25)
    logging.info("Overall median acc: {:.2f}%".format(np.median(overall_im_class_acc)*100))
    logging.info("Overall mean acc: {:.2f}%".format(np.mean(overall_im_class_acc)*100))
    logging.info("Overall std acc: {:.2f}%".format(np.std(overall_im_class_acc)*100))
    logging.info("")
    logging.info("classification CM - 4 class: \n{}".format(confusion_matrix(test_gt,test_preds)))
    logging.info("\n Report on test only \n"+classification_report(test_gt,test_preds, target_names=classes))
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
    logging.info("Overall mean MCC: {:.2f}%".format(np.mean(overall_im_class_mcc)*100))
    logging.info("Overall std MCC: {:.2f}%".format(np.std(overall_im_class_mcc)*100))

    if not opt.validate:
        best_validation_accuracy = np.array(best_validation_accuracy)
        best_val_error = np.array(best_val_error)
        logging.info("Segment-level")
        logging.info("median acc: {:.5f}".format(np.median(best_validation_accuracy)))
        logging.info("mean acc: {:.5f}".format(np.mean(best_validation_accuracy)))
        logging.info("std acc: {:.5f}".format(np.std(best_validation_accuracy)))
        logging.info("")
        logging.info("median error: {:.5f}".format(np.median(best_val_error)))
        logging.info("mean error: {:.5f}".format(np.mean(best_val_error)))
        logging.info("std error: {:.5f}".format(np.std(best_val_error)))
