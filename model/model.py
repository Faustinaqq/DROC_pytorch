# import torch
import torch.nn as nn
from model.resnet import ResNet18, ResNet34, ResNet50
# from torchvision.models import resnet18

class SimModel(nn.Module):
	def __init__(self, net: str, rep_dim=512, in_channels=3, dims=None, num_class=2):
		super(SimModel, self).__init__()
		self.net = None
		self.rep_dim = rep_dim
		# self.n_components = g_n
		# self.eps = 1e-06
		if net == "resnet18":
			self.net = ResNet18(in_channels, rep_dim)
			# self.net = resnet18(pretrained=True)
		elif net == "resnet34":
			self.net = ResNet34(in_channels, rep_dim)
		elif net == "resnet50":
			self.net = ResNet50(in_channels, rep_dim) 
			
		self.get_head(rep_dim, dims, num_class)

	def get_head(self, in_dim, dims=None, num_class=2):
		heads = ()
		if dims is not None:
			for i, d in enumerate(dims[:-1]):
				heads= heads + (nn.Linear(in_dim, d, bias=False),
                               nn.BatchNorm1d(d),
                               nn.ReLU())
				in_dim = d
			heads = heads + (nn.Linear(in_dim, dims[-1], bias=num_class > 0), )
		self.head = nn.Sequential(*heads)
		self.last_head = nn.Linear(dims[-1], num_class)
		self.num_class = num_class
  
	def forward(self, data):
		# print("data.size: ", data.size())
		pools = self.net(data)
		# print("pools: ", pools.size())
		embeds = self.head(pools)
		# print("embeds: ", embeds.size())
		logits = self.last_head(embeds)
		# print("logits: ", logits.size())
		return {'logits': logits, 'embeds': embeds, 'pools': pools}

