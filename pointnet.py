import torch
import torch.nn as nn

class TNet(nn.Module):

	def __init__(self, k=3):

		super(TNet, self).__init__()

		self.k = k
		self.conv1 = nn.Conv1d(k, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, k*k)
		self.relu = nn.ReLU()

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)
		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)

	def forward(self, x):

		batch_size = x.size()[0]
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.relu(self.bn2(self.conv2(x)))
		x = self.relu(self.bn3(self.conv3(x)))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)

		x = self.relu(self.bn4(self.fc1(x)))
		x = self.relu(self.bn5(self.fc2(x)))
		x = self.fc3(x)

		iden = torch.eye(self.k).view(1, self.k * self.k).repeat(batch_size, 1)

		if x.is_cuda:

			iden = iden.cuda()

		x = x + iden
		x = x.view(-1, self.k, self.k)

		return x

class PointNet(nn.Module):

	def __init__(self, num_classes=3, feature_transform=True):

		super(PointNet, self).__init__()

		self.feature_transform = feature_transform
		self.stn = TNet(k=3)
		self.conv1 = nn.Conv1d(3, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)

		if self.feature_transform:

			self.fstn = TNet(k=64)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, num_classes)
		self.dropout = nn.Dropout(p=0.3)
		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)
		self.relu = nn.ReLU()

	def forward(self, x):

		trans = self.stn(x)
		x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
		x = self.relu(self.bn1(self.conv1(x)))

		if self.feature_transform:

			trans_feat = self.fstn(x)
			x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)

		else:

			trans_feat = None

		x = self.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))

		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)

		x = self.relu(self.bn4(self.fc1(x)))
		x = self.relu(self.bn5(self.fc2(x)))
		x = self.dropout(x)
		x = self.fc3(x)

		return x, trans_feat

def feature_transform_regularizer(trans):

	d = trans.size()[1]
	I = torch.eye(d).cuda() if trans.is_cuda else torch.eye(d)
	loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))

	return loss