import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.special import expit



# Training Options
#==============================================================================================================
Data = 'MNIST'         # MNIST dataset
# Data = 'FMNIST'      # Fashion-MNIST dataset
EPOCH = 10             # Training epoch
batch = 100            # mini-batch size
H = 30                # the number of nodes in the hidden layer (Z)
lr = 0.005             # learning rate
view = -1              # the snapshot time(epoch) of the visualization of the cluster 
                       # default : -1 (the last epoch) (0 <= view < EPOCH) 
sample = 15            # the number of images showing
epsilon = 10**(-8)     # divergence regulator
DEVICE = "cpu"
#==============================================================================================================

# Data Load
#==============================================================================================================
if Data == 'MNIST':
    trainset = datasets.MNIST(root = './.data/', train = True, download = True, transform = transforms.ToTensor())
    testset = datasets.MNIST(root = './.data/', train = False, download = True, transform = transforms.ToTensor())
elif Data == 'FMNIST':
    trainset = datasets.FashionMNIST(root = './.data/', train = True, download = True, transform = transforms.ToTensor())
    testset = datasets.FashionMNIST(root = './.data/', train = False, download = True, transform = transforms.ToTensor())    

train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size=batch, shuffle = True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size=batch, shuffle = True, num_workers=0)
#==============================================================================================================


class RBM(nn.Module):
    def __init__(self, n_vis, n_hin, k):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
    def sample_from_p(self,p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
    def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
    def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
    def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)       
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)        
        return pre_h_, h_, v, pre_v_, v_

    def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        zr = Variable(torch.zeros(wx_b.size()))
        mask = torch.max(zr, wx_b)
        hidden_term = (((wx_b - mask).exp() + (-mask).exp()).log() + (mask)).sum(1)
        return (-hidden_term - vbias_term).mean()

model = RBM(784,H,1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

def train(model, train_loader, history_trainloss, history_testloss):
    model.train()
    for step, (data,target) in enumerate(train_loader):
        data = Variable(data.view(-1,784))
        x_train = data.bernoulli()

        pre_h, post_h, v, pre_v, post_v = model(x_train)
        train_loss = model.free_energy(v) - model.free_energy(post_v)
        history_trainloss.append(train_loss.data.detach().numpy())
        
        x_test = testset.data.view(-1,784).type(torch.FloatTensor)/255.
        _, _, v_test, _, post_v_test = model(x_test)
        test_loss = model.free_energy(v_test) - model.free_energy(post_v_test)
        history_testloss.append(test_loss.data.detach().numpy())
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    return (history_trainloss, history_testloss)


def Cluster(history_Z, view):
    if view == -1:
        i = -1
    else:
        i = np.int(view*len(hisotry_trainloss)/batch)
    bina_Z = history_Z[i]
    name_Z, count_Z = np.unique(bina_Z, return_counts=True, axis=0)
    k_Z, m_k_Z = np.unique(count_Z, return_counts=True)
    return (k_Z, m_k_Z)

def main():
    history_Z = []
    history_trainloss = []
    history_testloss = []
    sample_data = trainset.data[:sample].view(-1,28*28)
    sample_data = sample_data.type(torch.FloatTensor)/255.
        
    print('Training Starts!')
    print('(Energy based model(X-Z), Data : {})'.format(Data))

    for epoch in range(1,EPOCH+1):

        X = trainset.data.view(-1,28*28)
        X = X.type(torch.FloatTensor)/255.
        _, Z, _, _, _ = model(X)
        history_Z.append(Z.detach().numpy())
    
        history_trainloss, history_testloss = \
        train(model, train_loader, history_trainloss, history_testloss)

        test_x = sample_data.to(DEVICE)
        _,_,_,_,output = model(test_x)

        f,a = plt.subplots(2,sample,figsize=(sample,2))
        print("[Epoch {}]".format(epoch))
        for i in range(sample):
            img = np.reshape(sample_data.data.numpy()[i], (28,28))
            a[0][i].imshow(img, cmap='gray')
            a[0][i].set_xticks(()); a[0][i].set_yticks(())

        for i in range(sample):
            img = np.reshape(output.to("cpu").data.numpy()[i],(28,28))
            a[1][i].imshow(img, cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.show()

    print('Training Ends!')

    print('Visualizing the cluster frequency of the hidden layer...')
    
    k_Z, m_k_Z = Cluster(history_Z, view)
    
    fig = plt.figure(figsize=(5,5))
    plt.plot(np.log(k_Z), np.log(m_k_Z), 'r.', label='$Z$')
    plt.xlabel(r'$\log \:k$', fontsize=13)
    plt.ylabel(r'$\log \:m(k)$', fontsize=13)
    plt.title('Cluster distribution', fontsize=13)
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.show()
    
    





    