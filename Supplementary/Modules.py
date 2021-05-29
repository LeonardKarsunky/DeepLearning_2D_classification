import torch
import math

torch.set_grad_enabled(False)

# -------------------------------------------------------------------------- #

# Superclass Module
class Module(object) :
    def __init__(self):
        super().__init__()
    
    def forward(self , *input):
        raise  NotImplementedError
        
    def backward(self , *gradwrtoutput):
        raise  NotImplementedError
        
    def param(self): # These are the layers of the network
        return  []

# -------------------------------------------------------------------------- #

# Class Layer that Linear, Relu, Dropout, Loss, etc.. will inherit
class Layer(Module):
    def __init__(self):
        super().__init__()
        self.dropout = False
        self.linear = False
        
    def is_dropout(self):
        self.dropout = True
    
    def is_linear(self):
        self.linear = True

# -------------------------------------------------------------------------- #

# Class Sequential that will define and build the architecture of the model
class Sequential(Module):
    def __init__(self, param, Loss):
        super().__init__()
        self.model = (param)
        self.loss = Loss
    
    def forward(self, x):
        for layer in self.model:
            x = layer.forward(x)
        return x
    
    def backward(self, output, target):
        grad = self.loss.backward(target, output)
        
        for layer in reversed(self.model):
            grad = layer.backward(grad)
        
        Loss = self.loss.forward(target, output)
        return Loss
    
    def Train(self):
        for layer in self.model:
            if layer.dropout:
                layer.Train()
        
    def Eval(self):
        for layer in self.model:
            if layer.dropout:
                layer.Eval()
    
    def lr_method(self, method, lr):
        for layer in self.model:
            if layer.linear:
                layer.change_lr_method(method, lr)

# -------------------------------------------------------------------------- #

# Class Linear that will be use for building a MLP
class Linear(Layer):
    def __init__(self, in_, out_):
        super().__init__()
        self.in_ = in_
        self.out_ = out_
        self.is_linear()
        self.lr = 0.005
        self.lr_method = 'constant'
        
        # Capture the term at each layer before the passage in the layer
        # and the activation function.
        self.x = torch.zeros(out_)
        
        # Initialization of Adam for weight and bias
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1.0e-8
        self.eta = 1.0e-1
        self.mw = torch.zeros(out_)
        self.mb = torch.zeros(out_)
        self.vw = 0.0
        self.vb = 0.0
        
        # Initialization of the weights and the bias
        param = 1. / math.sqrt(in_)
        self.weight = torch.empty(self.in_, self.out_).uniform_(-param, param)
        self.bias = torch.empty(self.out_).uniform_(-param, param)
        
    def forward(self, x):
        self.x = x
        return x.mm(self.weight) + self.bias
    
    def set_Lr(self, lr):
        self.lr = lr
        return
        
    def backward(self, grad):
        
        if self.lr_method == "Adam":
            
            # Adam method for the learning rate
            gw = self.x.t().mm(grad)
            self.mw = ((self.beta1 * self.mw) + ((1 - self.beta1) * gw))
            mh = (1 / (1 - self.beta1)) * self.mw
            self.vw = ((self.beta2 * self.vw) + ((1 - self.beta2) * (gw.norm()**2)))
            vh = (1 / (1 - self.beta2)) * self.vw
            self.weight = self.weight - ((self.eta / (vh.sqrt() + self.eps)) * mh)

            self.mb = ((self.beta1 * self.mb) + ((1 - self.beta1) * grad))
            mh = (1 / (1 - self.beta1)) * self.mb
            self.vb = ((self.beta2 * self.vb) + ((1 - self.beta2) * (grad.norm()**2)))
            vh = (1 / (1 - self.beta2)) * self.vb
            self.bias = self.bias - ((self.eta / (vh.sqrt() + self.eps)) * mh)
            grad = grad.mm(self.weight.t())
            
        elif self.lr_method == "constant":
            
            # Constant learning rate
            self.weight = self.weight - self.lr * self.x.t().mm(grad)
            self.bias = self.bias - self.lr * grad * 1
            grad = grad.mm(self.weight.t())
            
        return grad
    
    def weight(self):
        return self.weight
    
    def bias(self):
        return self.bias
    
    def change_lr_method(self, method, lr):
        self.lr = lr
        self.lr_method = method

# -------------------------------------------------------------------------- #

# Class dropout that will be call when we want a dropout in our network
class Dropout(Layer) :
    def __init__(self):
        super().__init__()
        self.p = 0.2
        self.is_dropout()
        self.train = True
        
    
    def forward(self, x):
        n = torch.ones(x.size())
        if self.train:
            n = n * (1 - self.p)
            n = torch.bernoulli(n)
            n = n / (1 - self.p)
        return x * n
        
    def backward(self, x):
        return x
    
    def Train(self):
        self.train = True
        
    def Eval(self):
        self.train = False

# -------------------------------------------------------------------------- #

# Class that define the loss function computed as MSE loss
class LossMSE(Layer):
    def __init__(self):
        super().__init__() 
    
    def forward(self, data_target, data_output):
        loss = (data_output - data_target).pow(2).sum()
        return loss
    
    def backward(self, data_target, data_output):
        dloss = 2 * (data_output - data_target)
        return dloss
    
    def is_MSE(self):
        return True

# -------------------------------------------------------------------------- #

# Class that define the loss function computed with Cross Entropy
class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, data_target, data_output):
        output = data_output.to(dtype=torch.float)
        target = data_target.resize_(data_target.size(0), 1)
        
        zer = torch.zeros(target.size()).int()
        target = torch.cat((target,zer), 1)
    
        first_column = torch.tensor([0])
        loss = output.gather(1,target).index_select(1,first_column).exp()
        
        # To avoid numerical error in the computation
        maxx = loss.max()
        
        loss = (loss + maxx) / (output.exp().sum(1) + maxx)
        loss = -(loss.log().mean())
        return loss
    
    def backward(self, data_target, data_output):
        # New version
        N = data_target.size(0)
        dloss = data_output.exp()
        dloss = dloss / dloss.sum(1).resize_(N,1)
        
        add = data_target-1
        add = torch.cat((add, -data_target), 1)
        dloss = (1/N) * (dloss + add)
        return dloss
    
    def is_MSE(self):
        return False

# -------------------------------------------------------------------------- #

# Class for the activaction function: ReLU
class ReLU(Layer):
    
    def __init__(self ):
        super().__init__()
        self.save = 0
        
    def forward(self, x):
        y = x.clamp(min = 0)
        self.save = x
        return y
    
    def backward(self, x):
        y = self.save > 0
        return y.float() * x
         
    def print(self):
        return

# -------------------------------------------------------------------------- #

# Class for the activaction function: Leaky ReLU
class Leaky_ReLU(Layer):
    
    def __init__(self ):
        super().__init__()
        self.s = 0
        self.alpha = 0.01
        
    def forward(self, x):
        y = torch.maximum(self.alpha * x, x)
        self.s = x
        return y
    
    def backward(self, x):
        y = ((self.s > 0) * (1 - self.alpha)) + self.alpha
        return y.float() * x
         
    def print(self):
        return

# -------------------------------------------------------------------------- #

# Class for the activaction function: ELU
class ELU(Layer):
    
    def __init__(self):
        super().__init__()
        self.s = 0
        self.alpha = 0.01
        
    def forward(self, x):
        y = ((x > 0).float() * x) + (0 >= x) * self.alpha * (torch.exp(x) - 1)
        self.s = x
        return y
    
    def backward(self, x):
        y = ((self.s > 0) * (1 - self.alpha * torch.exp(self.s))) + self.alpha * torch.exp(self.s)
        return y.float() * x

# -------------------------------------------------------------------------- #

# Class for the activaction function: Tanh
class Tanh(Layer) :
    def __init__(self, ):
        super().__init__()
        self.save = 0
    
    def  forward(self, x):
        self.save = x
        return torch.div(x.exp() - (-x).exp(), x.exp() + (-x).exp())
        
    def  backward(self, x):
        return (1 - torch.div(self.save.exp() - 
                    (-self.save).exp(), self.save.exp() + (-self.save).exp())**2) * x
        
    def print(self):
        return

# -------------------------------------------------------------------------- #

# Class for the activaction function: Sigmoïd
class Sigmoid(Layer):
    
    def __init__(self):
        super().__init__()
        self.s = 0
        self.lambd = 3
        
    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.lambd * x))
        self.s = x
        return y
    
    def backward(self, x):
        y = self.lambd * torch.exp(-self.s) / ((1 + torch.exp(-self.lambd * self.s))**2)
        return y.float() * x  