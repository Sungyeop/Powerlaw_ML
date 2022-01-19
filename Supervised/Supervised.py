import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
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
n1 = 70                # the number of nodes in the first hidden layer (Z1)
n2 = 50                # the number of nodes in the second hidden layer (Z2)
n3 = 35                # the number of nodes in the third hidden layer (Z3)
lr = 0.005             # learning rate
activation = 'Sigmoid' # Sigmoid activation function
# activation = 'ReLU'  # ReLU activation function
view = -1              # the snapshot time(epoch) of the visualization of the cluster 
                       # default : -1 (the last epoch) (0 <= view < EPOCH) 
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


class MLP(nn.Module):
    
    def __init__(self, n1, n2, n3):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(28*28,n1)
        self.fc2 = nn.Linear(n1,n2)
        self.fc3 = nn.Linear(n2,n3)
        self.fc4 = nn.Linear(n3,10)

    def forward(self,x):
        x = x.view(-1, 784)
        if activation == 'Sigmoid':
            Z1 = torch.sigmoid(self.fc1(x))
            Z2 = torch.sigmoid(self.fc2(Z1))
            Z3 = torch.sigmoid(self.fc3(Z2))
        elif activation == 'ReLU':
            Z1 = torch.relu(self.fc1(x))
            Z2 = torch.relu(self.fc2(Z1))
            Z3 = torch.relu(self.fc3(Z2))
        Y = self.fc4(Z3)
        return Z1, Z2, Z3, Y

model = MLP(n1,n2,n3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            _, _, _, output = model(data)

            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def train(model, train_loader, history_W1, history_b1, history_W2, history_b2, history_W3, history_b3, \
          history_W4, history_b4, history_trainloss, history_testloss): 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        _, _, _, output = model(data)
        
        W1 = model.fc1.weight.data.detach().numpy()
        b1 = model.fc1.bias.data.detach().numpy()
        W2 = model.fc2.weight.data.detach().numpy()
        b2 = model.fc2.bias.data.detach().numpy()
        W3 = model.fc3.weight.data.detach().numpy()
        b3 = model.fc3.bias.data.detach().numpy()
        W4 = model.fc4.weight.data.detach().numpy()
        b4 = model.fc4.bias.data.detach().numpy()
        
        history_W1.append(copy.deepcopy(W1))
        history_b1.append(copy.deepcopy(b1))
        history_W2.append(copy.deepcopy(W2))
        history_b2.append(copy.deepcopy(b2))  
        history_W3.append(copy.deepcopy(W3))
        history_b3.append(copy.deepcopy(b3))  
        history_W4.append(copy.deepcopy(W4))
        history_b4.append(copy.deepcopy(b4))  
        
        train_loss = F.cross_entropy(output, target)
        history_trainloss.append(train_loss.detach().numpy())
        test_data = testset.data.view(-1,784).type(torch.FloatTensor)/255.
        Y_test = testset.targets
        _, _, _, Y_test_pred = model(test_data)
        test_loss = F.cross_entropy(Y_test_pred, Y_test)
        history_testloss.append(test_loss.detach().numpy())        
        
        train_loss.backward()
        optimizer.step()
        
    return (history_W1, history_b1, history_W2, history_b2, history_W3, history_b3, \
            history_W4, history_b4, history_trainloss, history_testloss)

def Sigmoid(x):
    return expit(x)

def ReLU(x):
    return x * (x>0)

def FF(test, W1, b1, W2, b2, W3, b3, W4, b4):
    if activation == 'Sigmoid':
        E1 = Sigmoid(np.einsum('ij,jk->ik', test, W1.T) + b1)
        E2 = Sigmoid(np.einsum('ij,jk->ik', E1, W2.T) + b2)
        E3 = Sigmoid(np.einsum('ij,jk->ik', E2, W3.T) + b3)
    elif activation == 'ReLU':
        E1 = ReLU(np.einsum('ij,jk->ik', test, W1.T) + b1)
        E2 = ReLU(np.einsum('ij,jk->ik', E1, W2.T) + b2)
        E3 = ReLU(np.einsum('ij,jk->ik', E2, W3.T) + b3)
    Y = np.einsum('ij,jk->ik', E3, W4.T) + b4
    return E1, E2, E3, Y

def Cluster(history_W1, history_b1, history_W2, history_b2, history_W3, history_b3, history_W4, history_b4, view):
    if view == -1:
        i = -1
    else:
        i = np.int(view*len(hisotry_W1)/batch)

    W1 = history_W1[i]
    b1 = history_b1[i]
    b1 = np.reshape(b1, (1,len(b1)))
    W2 = history_W2[i]
    b2 = history_b2[i]
    b2 = np.reshape(b2, (1,len(b2)))
    W3 = history_W3[i]
    b3 = history_b3[i]
    b3 = np.reshape(b3, (1,len(b3)))
    W4 = history_W4[i]
    b4 = history_b4[i]
    b4 = np.reshape(b4, (1,len(b4)))
    
    X = trainset.data.view(-1,28*28)
    X = X.type(torch.FloatTensor)/255.
    X = X.detach().numpy()
    Z1, Z2, Z3, Y = FF(X, W1, b1, W2, b2, W3, b3, W4, b4)

    if activation == 'Sigmoid':
        bina_Z1 = np.where(Z1 > 0.5, 1,0)
        name_Z1, count_Z1 = np.unique(bina_Z1, return_counts=True, axis=0)
        k_Z1, m_k_Z1 = np.unique(count_Z1, return_counts=True)

        bina_Z2 = np.where(Z2 > 0.5, 1,0)
        name_Z2, count_Z2 = np.unique(bina_Z2, return_counts=True, axis=0)
        k_Z2, m_k_Z2 = np.unique(count_Z2, return_counts=True)

        bina_Z3 = np.where(Z3 > 0.5, 1,0)
        name_Z3, count_Z3 = np.unique(bina_Z3, return_counts=True, axis=0)
        k_Z3, m_k_Z3 = np.unique(count_Z3, return_counts=True)
    elif activation == 'ReLU':
        bina_Z1 = np.where(Z1 > np.mean(Z1), 1,0)
        name_Z1, count_Z1 = np.unique(bina_Z1, return_counts=True, axis=0)
        k_Z1, m_k_Z1 = np.unique(count_Z1, return_counts=True)

        bina_Z2 = np.where(Z2 > np.mean(Z2), 1,0)
        name_Z2, count_Z2 = np.unique(bina_Z2, return_counts=True, axis=0)
        k_Z2, m_k_Z2 = np.unique(count_Z2, return_counts=True)

        bina_Z3 = np.where(Z3 > np.mean(Z3), 1,0)
        name_Z3, count_Z3 = np.unique(bina_Z3, return_counts=True, axis=0)
        k_Z3, m_k_Z3 = np.unique(count_Z3, return_counts=True)
    
    return (k_Z1, m_k_Z1, k_Z2, m_k_Z2, k_Z3, m_k_Z3)


def main():
    history_W1 = []
    history_b1 = []
    history_W2 = []
    history_b2 = []
    history_W3 = []
    history_b3 = []
    history_W4 = []
    history_b4 = []
    history_W5 = []
    history_b5 = []
    history_W6 = []
    history_b6 = []
    history_trainloss = []
    history_testloss = []
        
    print('Training Starts!')
    print('(Classification model(X-Z1-Z2-Z3-Y), Data : {}, Activation : {})'.format(Data, activation))
    
    for epoch in range(1, EPOCH + 1):
        history_W1, history_b1, history_W2, history_b2, history_W3, history_b3, history_W4, history_b4, history_trainloss, history_testloss = \
        train(model, train_loader, history_W1, history_b1, history_W2, history_b2, history_W3, history_b3, history_W4, history_b4, history_trainloss, history_testloss)

        train_loss, train_accuracy = evaluate(model, train_loader)
        test_loss, test_accuracy = evaluate(model, test_loader)

        print('[{} epoch] Train accuracy: {:.2f}%, Test accuracy: {:.2f}%'.format(epoch, train_accuracy, test_accuracy))

    print('Training Ends!')

    print('Visualizing the cluster frequency of the hidden layer...')
    
    k_Z1, m_k_Z1, k_Z2, m_k_Z2, k_Z3, m_k_Z3 = \
    Cluster(history_W1, history_b1, history_W2, history_b2, history_W3, history_b3, history_W4, history_b4, view)

    fig = plt.figure(figsize=(5,5))
    plt.plot(np.log(k_Z1), np.log(m_k_Z1), 'r.', label='$Z_1$')
    plt.plot(np.log(k_Z2), np.log(m_k_Z2), 'g.', label='$Z_2$')
    plt.plot(np.log(k_Z3), np.log(m_k_Z3), 'b.', label='$Z_3$')
    plt.xlabel(r'$\log \:k$', fontsize=13)
    plt.ylabel(r'$\log \:m(k)$', fontsize=13)
    plt.title('Cluster distribution', fontsize=13)
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.show()
    
    





    
