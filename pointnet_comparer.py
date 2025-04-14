import os
import sys
import time
import numpy as np
import torch
import csv
import tracemalloc
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
import copy
import warnings
import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report

from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules
from mmengine import init_default_scope
from mmengine.registry import build_from_cfg

from pointnet import PointNet
from pointnet_pp import PointNetPlusPlus

warnings.filterwarnings("ignore")

class ModelComparer:

	model_memory_usages = []

	def __init__(self, checkpoint_files, device='cpu'):
		"""
		Initializes the model comparer.

		Args:
			checkpoint_files (list): List of paths to checkpoint files.
			device (str): Device for inference ('cuda:0' or 'cpu').
		"""

		self.device = device
		self.models = []
		self.model_names = []
		self.classes = ['Car', 'Pedestrian', 'Cyclist']

		for checkpoint_file in checkpoint_files:

			model_name = os.path.basename(checkpoint_file).split('_')[0]

			if model_name == 'pointnet':

				self.model_names.append('pointnet')

				tracemalloc.start()

				new_model = PointNet(num_classes=len(self.classes))

				checkpoint = torch.load(checkpoint_file, map_location=self.device)
				new_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

				new_model = new_model.to(self.device)
				new_model.eval()

				_, peak_memory = tracemalloc.get_traced_memory()
				tracemalloc.stop()

				self.model_memory_usages.append(peak_memory / 10 ** 6)

				self.models.append(new_model)

			elif model_name == 'pointnetpp':

				self.model_names.append('pointnetpp')

				tracemalloc.start()

				new_model = PointNetPlusPlus(num_classes=len(self.classes))

				checkpoint = torch.load(checkpoint_file, map_location=self.device)
				new_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

				new_model = new_model.to(self.device)
				new_model.eval()

				_, peak_memory = tracemalloc.get_traced_memory()
				tracemalloc.stop()

				self.model_memory_usages.append(peak_memory / 10 ** 6)

				self.models.append(new_model)

			else:

				raise ValueError(f"Unsupported model type: {model_name}")

		self.results = {}

	def prepare_dataset(self, dataset_config):
		"""
		Prepares the dataset for evaluation.

		Args:
			dataset_config (dict): Dataset configuration.

		Returns:
			dataset: Evaluation dataset.
		"""

		dataset = build_from_cfg(dataset_config, DATASETS)
		return dataset

	@torch.no_grad()
	def evaluate_model(self, model, model_name, dataset, num_samples=None):
		"""
		Evaluates a model on a dataset using ground truth annotations.

		Args:
			model: The model to evaluate.
			model_name: Name of the model being evaluated.
			dataset: The dataset to evaluate on.
			num_samples (int, optional): Number of samples to evaluate. If None, evaluates the whole dataset.

		Returns:
			dict: Evaluation results with extended metrics.
		"""

		if num_samples is None:

			num_samples = len(dataset)

		else:

			num_samples = min(num_samples, len(dataset))

		results = []
		gt_labels = []
		inference_times = []			# Per scene (sample)
		inference_times_single = []		# Per object within scene
		memory_usages = []				# Per scene (sample)
		memory_usages_single = []		# Per object within scene

		for i in tqdm.tqdm(range(num_samples), desc=f"Evaluating {model_name}", unit="sample"):

			try:

				tracemalloc.start()

				data_info = copy.deepcopy(dataset[i])

				points = data_info['inputs']['points']
				points_np = points.cpu().numpy()

				gt_bboxes_3d = data_info['data_samples'].eval_ann_info['gt_bboxes_3d']
				gt_labels_3d = data_info['data_samples'].eval_ann_info['gt_labels_3d']

				o3d_cloud = o3d.geometry.PointCloud()
				o3d_cloud.points = o3d.utility.Vector3dVector(points_np[:, :3])

				scene_times = []
				item_pred_labels = []

				for box_idx, bbox in enumerate(gt_bboxes_3d.tensor.cpu().numpy()):

					center = bbox[:3]
					dimensions = bbox[3:6]
					yaw = bbox[6]

					adjusted_center = center.copy()
					adjusted_center[2] += dimensions[2] / 2

					rotation_matrix = np.array([
						[np.cos(yaw), -np.sin(yaw), 0],
						[np.sin(yaw), np.cos(yaw), 0],
						[0, 0, 1]
					])

					box = o3d.geometry.OrientedBoundingBox(
						adjusted_center,
						rotation_matrix,
						dimensions[[0, 1, 2]]
					)

					indices = box.get_point_indices_within_bounding_box(o3d_cloud.points)

					if len(indices) < 32:

						gt_labels_3d[box_idx] = -1

						continue

					cluster_points = points_np[indices, :3]

					cluster_mean = np.mean(cluster_points, axis=0)
					cluster_points = cluster_points - cluster_mean

					max_abs = np.max(np.abs(cluster_points))
					cluster_points = cluster_points / max_abs

					if len(cluster_points) > 1024:

						idx_choice = np.random.choice(len(cluster_points), 1024, replace=False)
						cluster_points = cluster_points[idx_choice]

					elif len(cluster_points) < 1024:

						idx_choice = np.random.choice(len(cluster_points), 1024 - len(cluster_points), replace=True)
						padding = cluster_points[idx_choice]
						cluster_points = np.vstack([cluster_points, padding])

					input_tensor = torch.FloatTensor(cluster_points).unsqueeze(0).transpose(1, 2).to(self.device)

					start_time = time.time()

					if model_name == 'pointnet':

						batch_results, _ = model(input_tensor)

					elif model_name == 'pointnetpp':

						batch_results = model(input_tensor)

					end_time = time.time()
					scene_times.append(end_time - start_time)

					_, predicted = torch.max(batch_results.data, 1)
					item_pred_labels.append(predicted.item())

				_, peak_memory = tracemalloc.get_traced_memory()
				tracemalloc.stop()

				del data_info

				if item_pred_labels:

					item_gt_labels = gt_labels_3d.tolist()
					item_gt_labels = [label for label in item_gt_labels if label != -1]

					assert len(item_pred_labels) == len(item_gt_labels), f"Length mismatch: {len(item_pred_labels)} vs {len(item_gt_labels)}"

					results.append(np.array(item_pred_labels))
					gt_labels.append(np.array(item_gt_labels))

					memory_usage = peak_memory / 10 ** 6

					memory_usages.append(memory_usage)
					inference_times.append(np.sum(scene_times))

					memory_usages_single.append(memory_usage / len(item_gt_labels))
					inference_times_single.append(np.mean(scene_times))

				gc.collect()

				if torch.cuda.is_available():

					torch.cuda.empty_cache()

			except Exception as e:

				print(f"Error processing sample {i}: {e}")

				tracemalloc.stop()

				continue

		if not results or not gt_labels:

			print("Not enough results for evaluation")
			return self._get_empty_metrics()

		try:

			y_true = np.concatenate(gt_labels)
			y_pred = np.concatenate(results)

			min_len = min(len(y_true), len(y_pred))
			y_true = y_true[:min_len]
			y_pred = y_pred[:min_len]

			if len(y_true) == 0:

				return self._get_empty_metrics()

			accuracy = accuracy_score(y_true, y_pred)
			precision_by_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(self.classes)))
			recall_by_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(self.classes)))
			f1_by_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(self.classes)))

			precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
			recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
			f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

			precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
			recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
			f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

			cm = confusion_matrix(y_true, y_pred, labels=range(len(self.classes)))

			plt.figure(figsize=(10, 8))
			sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes, linewidths=0.5, linecolor='gray', cbar=True, square=True)

			plt.title('Confusion Matrix', fontsize=16)
			plt.xlabel('Predicted Label', fontsize=12)
			plt.ylabel('True Label', fontsize=12)
			plt.xticks(rotation=45, ha='right')
			plt.yticks(rotation=0)
			plt.tight_layout()

			cbar = plt.gca().collections[0].colorbar
			cbar.set_label('Number of samples', labelpad=15)

			plt.savefig(os.path.join('./output', f"{model_name}_confusion_matrix.png"))
			plt.close()

			print("\nDetailed metrics by class:")
			print(classification_report(y_true, y_pred, target_names=self.classes, digits=4, zero_division=0))

		except Exception as e:

			print(f"Error calculating metrics: {e}")
			return self._get_empty_metrics()

		avg_inference_time = np.mean(inference_times)
		std_inference_time = np.std(inference_times)
		avg_inference_time_single = np.mean(inference_times_single)
		std_inference_time_single = np.std(inference_times_single)

		avg_memory_usage = np.mean(memory_usages)
		std_memory_usage = np.std(memory_usages)
		avg_memory_usage_single = np.mean(memory_usages_single)
		std_memory_usage_single = np.std(memory_usages_single)

		model_index = self.model_names.index(model_name)

		metrics = {
			'accuracy': accuracy,
			'precision_macro': precision_macro,
			'recall_macro': recall_macro,
			'f1_macro': f1_macro,
			'precision_weighted': precision_weighted,
			'recall_weighted': recall_weighted,
			'f1_weighted': f1_weighted,
			'memory_usage': avg_memory_usage,
			'memory_usage_std': std_memory_usage,
			'memory_usage_single': avg_memory_usage_single,
			'memory_usage_single_std': std_memory_usage_single,
			'inference_time': avg_inference_time,
			'inference_time_std': std_inference_time,
			'inference_time_single': avg_inference_time_single,
			'inference_time_single_std': std_inference_time_single,
			'num_samples': len(np.concatenate(gt_labels)),
			'num_classes': len(self.classes),
			'model_memory_usage': self.model_memory_usages[model_index],
		}

		for i, cls in enumerate(self.classes):
			metrics[f'{cls}_precision'] = precision_by_class[i]
			metrics[f'{cls}_recall'] = recall_by_class[i]
			metrics[f'{cls}_f1'] = f1_by_class[i]

		print(f"\nGlobal accuracy: {accuracy:.4f}")
		print("\nMetrics by class:")

		for i, cls in enumerate(self.classes):
			print(f"{cls} - Precision: {precision_by_class[i]:.4f}, Recall: {recall_by_class[i]:.4f}, F1: {f1_by_class[i]:.4f}")

		print("\nMacro Metrics:")
		print(f"Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
		print("\nWeighted Metrics:")
		print(f"Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
		print(f"Memory usage (MB): {avg_memory_usage:.4f} ± {std_memory_usage:.4f}")
		print(f"Inference time (s): {avg_inference_time:.4f} ± {std_inference_time:.4f}")
		print(f"Memory usage per sample (MB): {avg_memory_usage_single:.4f} ± {std_memory_usage_single:.4f}")
		print(f"Inference time per sample (s): {avg_inference_time_single:.4f} ± {std_inference_time_single:.4f}")

		return metrics

	def _get_empty_metrics(self):
		"""
		Returns an empty metrics dictionary when evaluation fails.
		"""

		empty_metrics = {
			'accuracy': 0,
			'precision_macro': 0,
			'recall_macro': 0,
			'f1_macro': 0,
			'precision_weighted': 0,
			'recall_weighted': 0,
			'f1_weighted': 0,
			'memory_usage': 0,
			'memory_usage_std': 0,
			'memory_usage_single': 0,
			'memory_usage_single_std': 0,
			'inference_time': 0,
			'inference_time_std': 0,
			'inference_time_single': 0,
			'inference_time_single_std': 0,
			'num_samples': 0,
			'num_classes': len(self.classes),
			'model_memory_usage': 0,
		}

		for cls in self.classes:
			empty_metrics[f'{cls}_precision'] = 0
			empty_metrics[f'{cls}_recall'] = 0
			empty_metrics[f'{cls}_f1'] = 0

		return empty_metrics

	def compare_models(self, dataset_config, num_samples=None, output_dir=None):
		"""
		Compares all models on the same dataset.

		Args:
			dataset_config (dict): Dataset configuration.
			num_samples (int, optional): Number of samples to evaluate.
			output_dir (str, optional): Directory to save results.

		Returns:
			dict: Evaluation results for all models.
		"""
		dataset = self.prepare_dataset(dataset_config)

		for i, model in enumerate(self.models):

			model_name = self.model_names[i]

			print(f"Evaluating model: {model_name}")

			self.results[model_name] = self.evaluate_model(model, model_name, dataset, num_samples)

			if output_dir:

				csv_path = os.path.join(output_dir, f"{model_name}_test_metrics.csv")
				save_metrics_csv(self.results[model_name], csv_path)

		return self.results

def save_metrics_csv(metrics_dict, filepath):
	"""
	Saves metrics to a CSV file.

	Args:
		metrics_dict (dict): Dictionary of metrics.
		filepath (str): Path to save the CSV file.
	"""

	os.makedirs(os.path.dirname(filepath), exist_ok=True)

	with open(filepath, 'w', newline='') as csvfile:

		fieldnames = list(metrics_dict.keys())
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		writer.writerow(metrics_dict)

	print(f"Metrics saved to {filepath}")

if __name__ == "__main__":

	init_default_scope('mmdet3d')
	register_all_modules()

	kitti_root = '/media/pablo/Disco programas/datasets/mmdetection3d/data/kitti/'

	if not os.path.exists(kitti_root):

		print(f"ERROR: The KITTI root directory does not exist: {kitti_root}")

		sys.exit(1)

	weights_path = './../pointnet/output/'
	checkpoint_files = [
		os.path.join(weights_path, "pointnet_best.pth"),
		os.path.join(weights_path, "pointnetpp_best.pth"),
	]

	output_dir = './output'
	os.makedirs(output_dir, exist_ok=True)

	dataset_config = dict(
		type='KittiDataset',
		data_root=kitti_root,
		ann_file=os.path.join(kitti_root, 'kitti_infos_val.pkl'),
		data_prefix=dict(
			pts='training/velodyne_reduced'
		),
		pipeline=[
			dict(
				type='LoadPointsFromFile',
				coord_type='LIDAR',
				load_dim=4,
				use_dim=4,
				backend_args=None
			),
			dict(
				type='MultiScaleFlipAug3D',
				img_scale=(1333, 800),
				pts_scale_ratio=1,
				flip=False,
				transforms=[
					dict(
						type='GlobalRotScaleTrans',
						rot_range=[0, 0],
						scale_ratio_range=[1., 1.],
						translation_std=[0, 0, 0]
					),
					dict(
						type='RandomFlip3D'
					),
					dict(
						type='PointsRangeFilter',
						point_cloud_range=[0, -40, -3, 70.4, 40, 1]
					)
				]
			),
			dict(
				type='Pack3DDetInputs', keys=['points'],
			)
		],
		modality=dict(use_lidar=True, use_camera=False),
		test_mode=True,
		metainfo=dict(classes=['Car', 'Pedestrian', 'Cyclist']),
		box_type_3d='LiDAR',
	)

	print(f"Using KITTI dataset: {kitti_root}")
	print(f"Annotations file: {os.path.join(kitti_root, 'kitti_infos_val.pkl')}")

	if not os.path.isfile(os.path.join(kitti_root, 'kitti_infos_val.pkl')):

		print("ERROR: The annotations file does not exist.")

		sys.exit(1)

	try:

		comparer = ModelComparer(checkpoint_files, device='cuda:0')

		results = comparer.compare_models(dataset_config, num_samples=None, output_dir=output_dir)

		print("\n===== Model comparison results =====")

		for model_name, metrics in results.items():

			print(f"\nResults for {model_name}:")

			for metric, value in metrics.items():

				print(f"\t{metric}: {value}")

	except Exception as e:

		print(f"ERROR: {e}")
