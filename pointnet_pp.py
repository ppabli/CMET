import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
	"""
	Calculate Euclideant distance between each two points.
	Input:
		src: source points, [B, N, C]
		dst: target points, [B, M, C]
	Return:
		dist: per-point square distance, [B, N, M]
	"""

	return torch.cdist(src, dst, p=2) ** 2

def index_points(points, idx):
	"""
	Input:
		points: input points data, [B, N, C]
		idx: sample index data, [B, S, [K]]
	Return:
		new_points:, indexed points data, [B, S, [K], C]
	"""

	B = points.shape[0]

	view_shape = list(idx.shape)

	view_shape[1:] = [1] * (len(view_shape) - 1)

	repeat_shape = list(idx.shape)

	repeat_shape[0] = 1

	batch_indices = torch.arange(B, dtype=torch.long).to(points.device).view(view_shape).repeat(repeat_shape)

	new_points = points[batch_indices, idx, :]

	return new_points

def farthest_point_sample(xyz, npoint):
	"""
	Input:
		xyz: pointcloud data, [B, N, 3]
		npoint: number of samples
	Return:
		centroids: sampled pointcloud index, [B, npoint]
	"""

	B, N, _ = xyz.shape

	centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
	distance = torch.full((B, N), 1e10, device=xyz.device)
	farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)

	for i in range(npoint):

		centroids[:, i] = farthest
		centroid = xyz[torch.arange(B), farthest].unsqueeze(1)
		dist = torch.sum((xyz - centroid) ** 2, dim=-1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = torch.max(distance, dim=1)[1]

	return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
	"""
	Input:
		radius: local region radius
		nsample: max sample number in local region
		xyz: all points, [B, N, 3]
		new_xyz: query points, [B, S, 3]
	Return:
		group_idx: grouped points index, [B, S, nsample]
	"""

	sqrdists = square_distance(new_xyz, xyz)
	group_idx = sqrdists.argsort()[:, :, :nsample]
	mask = sqrdists.gather(2, group_idx) > radius ** 2
	group_first = group_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, nsample)
	group_idx[mask] = group_first[mask]

	return group_idx

class SetAbstraction(nn.Module):

	def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):

		super().__init__()

		self.npoint = npoint
		self.radius = radius
		self.nsample = nsample
		self.group_all = group_all

		last_channel = in_channel + 3 if in_channel > 0 else 3
		self.mlp = nn.Sequential()

		for i, out_channel in enumerate(mlp):

			self.mlp.add_module(f'conv{i}', nn.Conv2d(last_channel, out_channel, 1))
			self.mlp.add_module(f'bn{i}', nn.BatchNorm2d(out_channel))
			self.mlp.add_module(f'relu{i}', nn.ReLU(inplace=True))
			last_channel = out_channel

	def forward(self, xyz, points):
		"""
		Input:
			xyz: input points position data, [B, N, 3]
			points: input points data, [B, N, D] or None
		Return:
			new_xyz: sampled points position data, [B, S, 3]
			new_points: sample points feature data, [B, S, D']
		"""

		B, _, C = xyz.shape

		if self.group_all:

			new_xyz = torch.zeros(B, 1, C, device=xyz.device)
			grouped_xyz = xyz.unsqueeze(1)

		else:

			fps_idx = farthest_point_sample(xyz, self.npoint)
			new_xyz = index_points(xyz, fps_idx)
			idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
			grouped_xyz = index_points(xyz, idx)

		grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

		if points is not None:

			grouped_points = index_points(points, idx) if not self.group_all else points.unsqueeze(1)
			grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)

		else:

			grouped_points = grouped_xyz_norm

		grouped_points = grouped_points.permute(0, 3, 2, 1)
		new_points = self.mlp(grouped_points).max(dim=2)[0].transpose(1, 2)

		return new_xyz, new_points

class PointNetPlusPlus(nn.Module):

	def __init__(self, num_classes=3, normal_channel=False):

		super().__init__()

		self.normal_channel = normal_channel

		in_channel = 3 if normal_channel else 0

		self.sa1 = SetAbstraction(512, 0.2, 32, in_channel, [64, 64, 128])
		self.sa2 = SetAbstraction(128, 0.4, 64, 128, [128, 128, 256])
		self.sa3 = SetAbstraction(None, None, None, 256, [256, 512, 1024], group_all=True)

		self.classifier = nn.Sequential(
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.4),
			nn.Linear(512, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.4),
			nn.Linear(256, num_classes)
		)

	def forward(self, xyz):

		B, _, _ = xyz.shape

		xyz = xyz.transpose(2, 1).contiguous()

		norm = xyz[:, :, 3:] if self.normal_channel else None
		xyz = xyz[:, :, :3]

		l1_xyz, l1_points = self.sa1(xyz, norm)
		l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
		_, l3_points = self.sa3(l2_xyz, l2_points)

		return self.classifier(l3_points.view(B, -1))