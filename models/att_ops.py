################################################
# Image-level classification Ops
################################################

import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import csv
import numpy as np
from utils.pytorchtools import EarlyStopping
from utils.utilities import initialize_logger
import os
import sys
import time
import logging
import copy
import math
from scipy import stats
from utils.visualize import *
from utils.utilities import CustomDataset
import cv2

def li_regularizer(net, loss):
    li_reg_loss = 0
    for m in net.modules():
        if isinstance(m,nn.Linear):
            temp_loss = torch.sum(((torch.sum(((m.weight)**2),1))**0.5),0)
            li_reg_loss += 0.1*temp_loss
    loss = loss + Variable(0.01*li_reg_loss, requires_grad= True)
    return loss

def train_model(model_num, model, data, criterion, optimizer, hist_dir, early_stop_patience, num_epochs=25, with_cuda=True):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_val_acc = 0
    best_epoch = 0

    hist_dir = os.path.join(hist_dir,'model_'+str(model_num))
    if not os.path.exists(hist_dir):
        os.mkdir(hist_dir)

    csv = open(os.path.join(hist_dir,'eval_history.csv'),'w')
    csv.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    
    early_stopping = EarlyStopping(patience=early_stop_patience, monitor='val_acc', verbose=False)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 12)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print(phase,'|',end = '')
            inputs, labels, patch_count, _ = data[phase]
            # Iterate over data.
            for i in range(len(patch_count)):
                #print(' ',i+1,') vs',inputs.size(1), end = '')
                input_im = inputs[i,:patch_count[i]]
                assert not torch.isnan(input_im).any()
                label_im = [labels[i].item()]
                label_im = torch.tensor(label_im)
                if torch.cuda.is_available() and with_cuda:
                    input_im = input_im.to('cuda')
                    label_im = label_im.to('cuda')
                # zero the parameter gradients
                optimizer.zero_grad()
                model._init_hidden_state(last_batch_size=1)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs,_,_ = model(input_im)
                    #assert not torch.isnan(outputs).any()
                    loss = criterion(outputs, label_im)
                    #loss = torch.sum(loss)
                    loss = li_regularizer(model,loss)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.50)
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == label_im.data)
            
            if phase == 'train':
                epoch_loss = running_loss / len(patch_count)
                epoch_acc = running_corrects.double() / len(patch_count)
            else:
                epoch_loss = running_loss / len(patch_count)
                epoch_acc = running_corrects.double() / len(patch_count)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                tr_loss = epoch_loss
                tr_acc = epoch_acc.cpu().data.numpy()
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc.cpu().data.numpy()

            # deep copy the model
            if phase == 'val' and val_acc > best_val_acc:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_acc = val_acc

                
            # save model
            #check_pt = "wt_ep{}.pth".format(epoch)
            #torch.save(model.state_dict(),os.path.join(hist_dir,check_pt))
        print()
                
        csv.write(str(epoch)+','+str(tr_loss)+','+str(tr_acc)+','+str(val_loss)+','+str(val_acc)+'\n')
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_acc, model,hist_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("best model epoch: {}, val_loss: {:.4f}, val_acc:{:.4f}".format(best_epoch, best_loss, best_val_acc))
    check_pt = "wt_best_ep{}_acc_{:.3f}.pth".format(best_epoch,best_val_acc)
    # save model
    model.load_state_dict(best_model_wts)
    
    # Best model predictions on the validation data 
    #writer.add_figure('predictions_vs._actuals', plot_classes_preds(model, inputs, labels, classes), global_step=model_num)

    torch.save(model.state_dict(),os.path.join(hist_dir,check_pt))
    return model, best_val_acc, best_loss, best_epoch

def infer(model, data, with_cuda=True,verbose=True):
    model.eval()   # Set model to evaluate mode
    overall_im_labels = []
    overall_im_probs = []
    inputs, labels, patch_count,im_names = data
    all_f = torch.tensor([])
    for i in range(len(patch_count)):
        input_im = inputs[i,:patch_count[i]]
        label_im = labels[i].item()
        if torch.cuda.is_available() and with_cuda:
            model = model.to('cuda')
            input_im = input_im.to('cuda')
        model._init_hidden_state(last_batch_size=1)
        output,output_att_wts_norm,att_wts = model(input_im)
        output_probs = torch.softmax(output,dim=1)
        max_value, pred_label = torch.max(output_probs, 1)
        if verbose:
            print("Im {} #{}, patch_count: {}".format(im_names[i],i,patch_count[i]))
            print("att_wts: ",att_wts.cpu().data.numpy())
            print(output_probs.cpu().data.numpy())
            print("best_prob: {}, pred_label: {}, gt:{}".format(max_value.item(),pred_label.item(),label_im))
        overall_im_labels.append(pred_label.item())
        overall_im_probs.append(output_probs.cpu().data.numpy())
        if not i:
            all_f = output
        else:
            all_f = torch.cat((all_f, output),dim=0)
    overall_im_labels = np.array(overall_im_labels)
    overall_im_probs = np.array(overall_im_probs)
    overall_gt = labels.numpy()
    all_l = labels
    return overall_im_labels, overall_im_probs, overall_gt, all_f,all_l 

def infer62(model, data, im_csv_list, gt_csv_list, with_cuda=True,verbose=True):
    model.eval()   # Set model to evaluate mode
    overall_im_labels = []
    overall_im_probs = []
    inputs, labels, patch_count,im_names = data

    for im_name in im_csv_list:
        idx = np.where(np.array(im_names) == im_name[:-4])
        if idx[0].shape[0]: # if there is a match shape[0] should 1, if not 0
            target.append(gt_csv_list[idx[0][0]])
        else:
            target.append(None)

    all_f = torch.tensor([])
    for i in range(len(patch_count)):
        input_im = inputs[i,:patch_count[i]]
        label_im = labels[i].item()
        if torch.cuda.is_available() and with_cuda:
            model = model.to('cuda')
            input_im = input_im.to('cuda')
        model._init_hidden_state(last_batch_size=1)
        output,output_att_wts_norm,att_wts = model(input_im)
        output_probs = torch.softmax(output,dim=1)
        max_value, pred_label = torch.max(output_probs, 1)
        if verbose:
            print("Im {} #{}, patch_count: {}".format(im_names[i],i,patch_count[i]))
            print("att_wts: ",att_wts.cpu().data.numpy())
            print(output_probs.cpu().data.numpy())
            print("best_prob: {}, pred_label: {}, gt:{}".format(max_value.item(),pred_label.item(),label_im))
        overall_im_labels.append(pred_label.item())
        overall_im_probs.append(output_probs.cpu().data.numpy())
        if not i:
            all_f = output
        else:
            all_f = torch.cat((all_f, output),dim=0)
    overall_im_labels = np.array(overall_im_labels)
    overall_im_probs = np.array(overall_im_probs)
    overall_gt = labels.numpy()
    all_l = labels
    return overall_im_labels, overall_im_probs, overall_gt, all_f,all_l 
