import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import sys
from utils.utilities import matrix_mul, element_wise_mul


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, bidirectional):
        super(BidirectionalLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.embedding = nn.Linear(nHidden * 2, nOut)
        else:
            self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input) # output has shape (seq_len, batch_size, hidden_dimensions) = w x nB x nIn
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        # nn.Linear operates on the last dimension of its input
        # i.e. for each slice [i, j, :] of rnn_output it produces a vector of size num_classes
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1) 
        return output, recurrent

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_classes=4):
        super(Attention, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        if output.dim() == 2:
            output = output.unsqueeze(1)
        output = matrix_mul(output, self.context_weight)
        if output.dim() == 1:
            output = output.unsqueeze(1)
        output = output.permute(1, 0)
        output = torch.softmax(output, dim=0)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)
        return output, h_output

class PretrainedNet(nn.Module):
    def __init__(self,c_in,model_name):
        super(PretrainedNet,self).__init__()
        if model_name == 'resnet101':
            model_ft = models.resnet101(pretrained=True)
        elif model_name == 'densenet121':
            model_ft = models.densenet121(pretrained=True)
        elif model_name == 'densenet161':
            model_ft = models.densenet161(pretrained=True)
        elif model_name == 'squeezenet':
            model_ft = models.squeezenet1_0(pretrained=True)
        elif model_name == 'resnext101_32x8d':
            model_ft = models.wide_resnet101_2(pretrained=True)
        elif model_name == 'inception_v3':
            model_ft = models.inception_v3(pretrained=True)

        self._set_parameter_requires_grad(model_ft)

        if model_name.startswith('res'):
            layers = [i for j, i in enumerate(model_ft.children()) if j < 7]
            self.model = nn.Sequential(*layers,nn.MaxPool2d((4,1)))
        elif model_name.startswith('dense'):
            layers = model_ft.features[:-3]
            self.model = nn.Sequential(*layers,nn.MaxPool2d((4,1)))
        elif model_name.startswith('squeeze'):
            layers = model_ft.features[:-2]
            self.model = nn.Sequential(*layers,nn.MaxPool2d((4,1)))
        elif model_name.startswith('incep'):
            layers = [i for j, i in enumerate(model_ft.children()) if j < 13]
            self.model = nn.Sequential(*layers,nn.MaxPool2d((8,4)))
        
    def _set_parameter_requires_grad(model, feature_extracting=True):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self,x):
        feature_map = self.model(x)
        return feature_map

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, batch_size, model_name=None, with_BLSTM=True, with_attention=False, cuda=True):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.batch_size = batch_size
        self.hidden_size = nh
        self.with_BLSTM = with_BLSTM
        self.with_attention = with_attention
        self.cuda = cuda
        # usual input size: 3x64x704
        self.cnn = PretrainedNet(nc, model_name)
        # cnn out size: nBatchxnum_fea_outx1x44
        if model_name == 'resnet18':
            num_fea_out = 256
        elif model_name == 'resnet50':
            num_fea_out = 512
        elif model_name == 'resnet101':
            num_fea_out = 1024
        elif model_name == 'densenet121':
            num_fea_out = 1024
        elif model_name == 'densenet161':
            num_fea_out = 2112
        elif model_name == 'squeezenet':
            num_fea_out = 512
        elif model_name == 'resnext101_32x8d':
            num_fea_out = 1024
        elif model_name == 'inception_v3':
            num_fea_out = 768

        if self.with_attention:
            self.rnn1 = BidirectionalLSTM(num_fea_out, nh, nh,bidirectional=True)
            self.rnn2 = BidirectionalLSTM(nh, nh, nh,bidirectional=True)
            self.attention = Attention(nh, nh, nclass)
            self._init_hidden_state()
        elif self.with_BLSTM:
            self.rnn1 = BidirectionalLSTM(num_fea_out, nh, nh,bidirectional=True)
            self.rnn2 = BidirectionalLSTM(nh, nh, nclass,bidirectional=True)
            #self.rnn = nn.Sequential(
            #BidirectionalLSTM(num_fea_out, nh, nh),
            #BidirectionalLSTM(nh, nh, nclass))
        else:
            self.fc = nn.Sequential(
                            nn.Linear(num_fea_out*44, 512),nn.ReLU(True),
                            nn.Linear(512, 256),nn.ReLU(True),
                            nn.Linear(256, 4)
                            )
    
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.hidden_state = torch.zeros(2, batch_size, self.hidden_size)
        if torch.cuda.is_available() and self.cuda:
            self.hidden_state = self.hidden_state.cuda()

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # [b, c, w]

        if self.with_attention:
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # rnn features
            output1,_ = self.rnn1(conv)
            output,recurrent = self.rnn2(output1)
            #output, self.hidden_state = self.attention(output, self.hidden_state)
            output, _ = self.attention(output, self.hidden_state)
            return output, output.view(b,-1),None
        elif self.with_BLSTM:
            conv = conv.permute(2, 0, 1).contiguous()  # [w, b, c]
            # rnn features
            output1,_ = self.rnn1(conv)
            output,recurrent = self.rnn2(output1)
            return output[-1], output[-1],recurrent
            #output = self.rnn(conv)
            #return output[-1], output.view(b,-1)
        else:
            conv = conv.view(-1,c*w)
            output = self.fc(conv)
            return output, output.view(b,-1),None
