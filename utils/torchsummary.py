import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
from torchvision import models

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, bidirectional,initial=False):
        super(BidirectionalLSTM, self).__init__()
        self.initial = initial
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.embedding = nn.Linear(nHidden * 2, nOut)
        else:
            self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        if self.initial:
            input = input.squeeze(2)
            input = input.permute(2, 0, 1).contiguous()
        recurrent, _ = self.rnn(input) # output has shape (seq_len, batch_size, hidden_dimensions) = w x nB x nIn
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        # nn.Linear operates on the last dimension of its input
        # i.e. for each slice [i, j, :] of rnn_output it produces a vector of size num_classes
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1) 
        return output, recurrent

def summary(model, input_size, batch_size=-1, device="cpu"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0][0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary

if __name__ == "__main__":
    """
    model_ft = models.densenet121(pretrained=True)
    model_ft = model_ft.to('cuda')
    layers = model_ft.features[:-3]
    model = nn.Sequential(*layers)
    model2 = nn.Sequential(model,
                            nn.Conv2d(1024,1024,(3,1)),
                            nn.BatchNorm2d(1024, momentum=0.01),
                            nn.Conv2d(1024,512,1),
                            nn.BatchNorm2d(512, momentum=0.01),
                            nn.ReLU(),
                            nn.Conv2d(512,512,(2,1)),
                            nn.BatchNorm2d(512, momentum=0.01),
                            nn.Conv2d(512,256,1),
                            nn.BatchNorm2d(256, momentum=0.01),
                            nn.ReLU())
    model2 = model2.to('cuda')
    summary(model2, (3,64,704))
    
    model_ft = models.inception_v3(pretrained=False)
    #model_ft = model_ft.to('cuda')
    layers = model_ft.features[:-2]
    model = nn.Sequential(*layers,
                            nn.MaxPool2d((2,1))) #512
    # model = nn.Sequential(*layers,
    #                         nn.MaxPool2d((4,1)),
    #                         BidirectionalLSTM(1024, 256, 256,bidirectional=True,initial=True),
    #                         BidirectionalLSTM(256, 256, 4,bidirectional=True))
    #print(model_ft)
    summary(model, (3,64,704))
    # print(len(layers))
    """
    model_ft = models.inception_v3(pretrained=False)
    #model_ft = model_ft.to('cuda')
    #summary(model_ft, (3, 64, 704))
    # print(model_ft.features[:-10]) # densenet
    layers = [i for j, i in enumerate(model_ft.children()) if j < 13]
    model = nn.Sequential(*layers,
                            nn.MaxPool2d((8,4)))
    #print(model_ft)
    summary(model, (3, 64, 704))
    

    