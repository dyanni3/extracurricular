import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F



def f(x, y, ep):
	return(np.sin(x + y**2) + ep*np.random.random(size=x.shape))

class MyDataset(Dataset):

	def __init__(self, datalist):
		self.datalist = datalist

	def __len__(self):
		return(len(self.datalist))

	def __getitem__(self, index):
		x0, x1 = self.datalist[index][0]
		y = self.datalist[index][1]
		X = torch.tensor([x0,x1])
		return X, y


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.dense1 = nn.Linear(in_features=2, out_features=50)
		self.dropout = nn.Dropout(p=.4)
		self.dense2 = nn.Linear(50, 50)
		self.out = nn.Linear(50, out_features=1)
		self.loss = nn.MSELoss(reduction='mean')
		self.optimizer = optim.Adam(self.parameters(), lr=2e-5)

	def forward(self, X):
		X = F.relu(self.dense1(X))
		X = self.dropout(X)
		X = F.relu(self.dense2(X))
		X = self.dropout(X)
		y = self.out(X)
		return(y)

	def train(self, train_loader, *args, **kwargs):
		#number_iterations = round(train_loader.__len__() / train_loader.batch_size)
		for (X, y) in tqdm(train_loader):
			self.optimizer.zero_grad()
			y_hat = self.forward(X)
			L = self.loss(y_hat, y)
			L.backward()
			self.optimizer.step()

	#def predict(self, test_loader, *args, **kwargs):
	#	self.eval()




if __name__ == '__main__':
	x = np.ogrid[-3:3:.01]
	y = np.ogrid[-3:3:.01]
	xx, yy = np.meshgrid(x,y)
	zz = f(xx, yy, 5e-1)
	plt.imshow(zz); plt.show()
	X = list(zip(xx.ravel(), yy.ravel()))
	y = zz.ravel()
	all_dat = list(zip(X,y))

	my_dat = MyDataset(all_dat)
	train, test = random_split(my_dat, [round(.7*my_dat.__len__()), round(.3*my_dat.__len__())])
	train_loader = DataLoader(my_dat, batch_size=32)
	test_loader = DataLoader(test, batch_size=64)

	# DataLoader just gets us nice pytorch batches, nothing more
	# to check out what DataLoader does try
	# data_iter = iter(train_loader)
	# X, y = data_iter.next()
	# print(X.shape, y.shape)

	net = Network()
	for ep in range(1000):
		print(f"episode: {ep+1}")
		net.train(train_loader)

	X = []
	y_hat = []
	y_true = []
	for _ in range(10_000):
		this_x_np = 6*(np.random.random(size=2)-.5)
		this_x = torch.tensor(list(this_x_np))
		this_y = net(this_x)
		this_y_np = this_y.data.numpy()[0]
		X.append(this_x_np)
		y_hat.append(this_y_np)
		y_true.append(f(*this_x_np, ep=0))

	X0 = np.array(X)[:,0]
	X1 = np.array(X)[:,1]

	with plt.style.context('dark_background'):
	    fig, ax = plt.subplots(ncols=2, figsize = (13,5))
	    ax[0].scatter(X0, X1, c=y_hat, cmap=plt.cm.viridis, s=.4, alpha=.85)
	    ax[1].scatter(X0, X1, c=y_true, cmap=plt.cm.viridis, s=.4, alpha=.85)
	    #ax[1].imshow(zz)
	    plt.show()