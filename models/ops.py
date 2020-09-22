################################################
# Segment-level sequence generator Ops
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

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init2(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        m.reset_parameters()

def weights_init3(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal(m.weight, mode='fan_out', a=0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)


#%% Train
def train_model(model_num, model, dataloaders, num_tr_samples, criterion, optimizer, hist_dir, early_stop_patience, writer, num_epochs=25, with_cuda=True):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_val_acc = 0
    best_epoch = 0
    all_f_ = torch.tensor([])
    all_l_ = torch.tensor([])

    hist_dir = os.path.join(hist_dir,'model_'+str(model_num))
    if not os.path.exists(hist_dir):
        os.mkdir(hist_dir)

    csv = open(os.path.join(hist_dir,'eval_history.csv'),'w')
    csv.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    
    monitor='val_acc'
    early_stopping = EarlyStopping(patience=early_stop_patience,monitor=monitor, verbose=False)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 12)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print(phase,'|',end = '')
            print(" Dataloader len: ",len(dataloaders[phase].dataset))
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                #print(' ',i+1,') vs',inputs.size(1), end = '')
                if torch.cuda.is_available() and with_cuda:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                # zero the parameter gradients
                optimizer.zero_grad()
                model._init_hidden_state(last_batch_size=inputs.size(0))

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs, features,_ = model(inputs)
                    loss = criterion(outputs, labels)
                    #loss = torch.sum(loss)
                    loss = li_regularizer(model,loss)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        if not i:
                            all_f = features
                            all_l = labels
                        else:
                            all_f = torch.cat((all_f, features),dim=0)
                            all_l = torch.cat((all_l, labels),dim=0)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                epoch_loss = running_loss / num_tr_samples
                epoch_acc = running_corrects.double() / num_tr_samples
            else:
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                tr_loss = epoch_loss
                tr_acc = epoch_acc.cpu().data.numpy()
                writer.add_scalar('loss/train', epoch_loss, epoch)
                writer.add_scalar('acc/train', epoch_acc, epoch)
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc.cpu().data.numpy()
                writer.add_scalar('loss/test', epoch_loss, epoch)
                writer.add_scalar('acc/test', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val':
                if monitor=='val_acc' and val_acc > best_val_acc: #epoch_loss < best_loss: 
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_val_acc = val_acc
                    all_f_ = all_f
                    all_l_ = all_l
                    
                elif monitor=='val_loss' and epoch_loss < best_loss: 
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_val_acc = val_acc
                    all_f_ = all_f
                    all_l_ = all_l

        print()    
        csv.write(str(epoch)+','+str(tr_loss)+','+str(tr_acc)+','+str(val_loss)+','+str(val_acc)+'\n')
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if monitor=='val_acc':
            early_stopping(val_acc, model,hist_dir)
        else:
            early_stopping(val_loss, model,hist_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))


    # save best predictions and respective labels
    #np.savez(os.path.join(hist_dir,'model_'+str(model_num)+'.npz'),
    #        pred=best_pred_out.cpu().data.numpy(),gt=best_gt.cpu().data.numpy())
    # load best model weights
    logging.info("best model epoch: {}, val_loss: {:.4f}, val_acc:{:.4f}".format(best_epoch, best_loss, best_val_acc))
    check_pt = "wt_best_ep{}_loss_{:.3f}_acc{:.3f}.pth".format(best_epoch,best_loss,best_val_acc)
    # save model
    model.load_state_dict(best_model_wts)
    
    # Best model predictions on the validation data 
    #writer.add_figure('predictions_vs._actuals', plot_classes_preds(model, inputs, labels, classes), global_step=model_num)

    torch.save(model.state_dict(),os.path.join(hist_dir,check_pt))
    return model, best_val_acc, best_loss, best_epoch, all_f_, all_l_


def infer(model_num, best_model, val_dataloader, classes, writer, with_cuda=True):
    # Validate models for PR-curves
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for images, labels in val_dataloader:
            if torch.cuda.is_available() and with_cuda:
                images = images.to('cuda')
                labels = labels.to('cuda')
            output,_,_ = best_model(images)
            class_probs_batch = [torch.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat([torch.stack(batch) for batch in class_preds])

    preds_on_last_batch = np.squeeze(class_preds_batch.cpu().numpy())
    probs_on_last_batch = [torch.softmax(el, dim=0)[i].item() for i, el in zip(preds_on_last_batch, output)]
    # Best model predictions on the validation data 
    writer.add_figure('predictions_vs._actuals', 
                        plot_classes_preds2(preds_on_last_batch, probs_on_last_batch, images, labels, classes), global_step=model_num)

    # plot all the pr curves
    for i in range(len(classes)):
        add_pr_curve_tensorboard(writer, i, test_probs, test_preds, classes, global_step=model_num)
    
    return test_preds.cpu().numpy()

def infer_max_vote(model_num, best_model, im_list, gt_list, im_DIR, classes, resize=False, cuda=True):
    best_model.eval()
    best_model._init_hidden_state(last_batch_size=1)
    overall_gt = np.array(gt_list)
    overall_im_labels = []
    transform = CustomDataset(phase='val').transform
    with torch.no_grad():
        for i,im_name in enumerate(im_list):
            preds_labels = []
            for im_patch_name in os.listdir(os.path.join(im_DIR,im_name)):
                if im_patch_name.endswith('tif'):
                    #img_as_PIL = Image.open(os.path.join(im_DIR,im_name,im_patch_name))
                    img_as_bgr = cv2.imread(os.path.join(im_DIR,im_name,im_patch_name))
                    if resize:
                        height = 704
                        width = 64
                        img_as_bgr = cv2.resize(img_as_bgr,(width,height), interpolation = cv2.INTER_CUBIC)
                    img_as_rgb = cv2.cvtColor(img_as_bgr, cv2.COLOR_BGR2RGB)
                    img_as_rgb = np.rot90(img_as_rgb)
                    augmented = transform(image=img_as_rgb)
                    img_as_tensor = augmented['image']
                    img_as_tensor = img_as_tensor.unsqueeze(0)
                    if torch.cuda.is_available() and cuda:
                        img_as_tensor = img_as_tensor.to('cuda')
                        best_model = best_model.to('cuda')
                    output,_,_ = best_model(img_as_tensor)
                    _, preds_label = torch.max(output, 1)
                    preds_labels.append(preds_label)
            preds_labels = np.array(preds_labels)
            overall_im_label,_ = stats.mode(preds_labels, axis=None)
            overall_im_labels.append(overall_im_label[0].item())
    overall_im_labels = np.array(overall_im_labels)
    return overall_im_labels, overall_gt 

def infer_avg_vote(model_num, best_model, im_list, gt_list, im_DIR, classes, resize=False, cuda=True):
    best_model.eval()
    best_model._init_hidden_state(last_batch_size=1)
    overall_gt = np.array(gt_list)
    overall_im_labels = []
    transform = CustomDataset(phase='val').transform
    with torch.no_grad():
        for i,im_name in enumerate(im_list):
            output_probs = []
            for im_patch_name in os.listdir(os.path.join(im_DIR,im_name)):
                if im_patch_name.endswith('tif'):
                    #img_as_PIL = Image.open(os.path.join(im_DIR,im_name,im_patch_name))
                    img_as_bgr = cv2.imread(os.path.join(im_DIR,im_name,im_patch_name))
                    if resize:
                        height = 704
                        width = 64
                        img_as_bgr = cv2.resize(img_as_bgr,(width,height), interpolation = cv2.INTER_CUBIC)
                    img_as_rgb = cv2.cvtColor(img_as_bgr, cv2.COLOR_BGR2RGB)
                    img_as_rgb = np.rot90(img_as_rgb)
                    augmented = transform(image=img_as_rgb)
                    img_as_tensor = augmented['image']
                    img_as_tensor = img_as_tensor.unsqueeze(0)
                    if torch.cuda.is_available() and cuda:
                        img_as_tensor = img_as_tensor.to('cuda')
                        best_model = best_model.to('cuda')
                    output,_,_ = best_model(img_as_tensor)
                    output_prob = torch.softmax(output,dim=1)
                    _, preds_label = torch.max(output, 1)
                    output_probs.append(output_prob.cpu().data.numpy())
            output_probs = np.array(output_probs)
            output_probs = output_probs.reshape(-1,4)
            overall_im_prob = np.mean(output_probs,axis=0)
            overall_im_label = np.argmax(overall_im_prob)
            print(overall_im_prob)
            print("pred_label: {}, gt:{}".format(overall_im_label,gt_list[i]))
            overall_im_labels.append(overall_im_label)
    overall_im_labels = np.array(overall_im_labels)
    return overall_im_labels, overall_gt 


def infer_feats(model_num, best_model, im_list, gt_list, im_DIR, classes,resize, cuda=True):
    best_model.eval()
    best_model._init_hidden_state(last_batch_size=1)
    overall_gt = np.array(gt_list)
    overall_im_labels = []
    transform = CustomDataset(phase='val').transform
    data_x_feats = torch.zeros(len(gt_list),128,4)
    im_patch_count = []
    with torch.no_grad():
        for i,im_name in enumerate(im_list):
            preds_labels = []
            patch_count = 0
            for im_patch_name in os.listdir(os.path.join(im_DIR,im_name)):
                if im_patch_name.endswith('tif'):
                    img_as_bgr = cv2.imread(os.path.join(im_DIR,im_name,im_patch_name))
                    if resize:
                        height = 704
                        width = 64
                        img_as_bgr = cv2.resize(img_as_bgr,(width,height), interpolation = cv2.INTER_CUBIC)
                    img_as_rgb = cv2.cvtColor(img_as_bgr, cv2.COLOR_BGR2RGB)
                    img_as_rgb = np.rot90(img_as_rgb)
                    augmented = transform(image=img_as_rgb)
                    img_as_tensor = augmented['image']
                    img_as_tensor = img_as_tensor.unsqueeze(0)
                    if torch.cuda.is_available() and cuda:
                        img_as_tensor = img_as_tensor.to('cuda')
                        best_model = best_model.to('cuda')
                    _,feats,_ = best_model(img_as_tensor)
                    data_x_feats[i][patch_count] = feats
                    patch_count += 1
            im_patch_count.append(patch_count)
    return data_x_feats, im_patch_count 

