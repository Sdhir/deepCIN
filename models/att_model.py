import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import sys
from utils.utilities import matrix_mul, element_wise_mul

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_classes=4):
        super(Attention, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(hidden_size, 1))

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        #print(f_output.cpu().data)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output[torch.isnan(output)] = 0
        if output.dim() == 2:
            output = output.unsqueeze(1)
        output = matrix_mul(output, self.context_weight)
        output[torch.isnan(output)] = 0
        if output.dim() == 1:
            output = output.unsqueeze(1)
        output = output.permute(1, 0)
        output_att_wts = torch.softmax(output, dim=1)
        output_att_wts_norm = output_att_wts
        #Normalize
        #output_att_wts_norm = (output_att_wts-torch.min(output_att_wts).item())/(torch.max(output_att_wts).item()-torch.min(output_att_wts).item())
        #Standardize
        #output_att_wts = (output_att_wts-torch.mean(output_att_wts))/torch.std(output_att_wts)
        #output_att_wts = torch.abs(output_att_wts)
        im_output = element_wise_mul(f_output, output_att_wts_norm.permute(1, 0)).squeeze(0)
        im_output = self.fc(im_output)
        return im_output,output_att_wts_norm, output_att_wts

class rnn_model(nn.Module):

    def __init__(self, nclass, nh_im, with_cuda=True):
        super(rnn_model, self).__init__()
        self.hidden_size = nh_im
        self.with_cuda = with_cuda
        self.batch_size=1
        self.attention = Attention(4, nh_im, nclass)
        self._init_hidden_state()
    
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.hidden_state = torch.zeros(1, batch_size, self.hidden_size)
        if torch.cuda.is_available() and self.with_cuda:
            self.hidden_state = self.hidden_state.cuda()

    def forward(self, input):
        #input [num_patches_in_im,1,4]
        input = input.view(-1,1,4)
        out_im = self.attention(input, self.hidden_state)
        return out_im
        