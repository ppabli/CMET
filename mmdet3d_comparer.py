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
import copy
import tqdm
import warnings

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.registry import DATASETS
from mmdet3d.utils import register_all_modules
from mmengine import init_default_scope
from mmengine.registry import build_from_cfg

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, classification_report

warnings.filterwarnings("ignore")

class ModelComparer:

	model_memory_usages = []

	def __init__(self, config_files, checkpoint_files, device='cpu'):
		"""
		Initializes the model comparer.

		Args:
			config_files (list): List of paths to config files.
			checkpoint_files (list): List of paths to checkpoint files.
			device (str): Device for inference ('cuda:0' or 'cpu').
		"""

		assert len(config_files) == len(checkpoint_files), "Number of config and checkpoint files must match"

		self.device = device
		self.models = []
		self.model_names = []

		for config_file, checkpoint_file in zip(config_files, checkpoint_files):

			tracemalloc.start()

			model = init_model(config_file, checkpoint_file, device=device)
			model.eval()

			_, peak = tracemalloc.get_traced_memory()
			tracemalloc.stop()

			self.model_memory_usages.append(peak / 10 ** 6)

			model_name = os.path.basename(config_file).split('_')[0]
			self.models.append(model)
			self.model_names.append(model_name)

		self.results = {}
		self.classes = ['Car', 'Pedestrian', 'Cyclist']

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
	def evaluate_model(self, model, model_name, dataset, num_samples=None, score_threshold=0.0):
		"""
		Evaluates a model on a dataset.

		Args:
			model: The model to evaluate.
			model_name: Name of the model being evaluated.
			dataset: The dataset to evaluate on.
			num_samples (int, optional): Number of samples to evaluate. If None, evaluates the whole dataset.
			score_threshold (float, optional): Score threshold for filtering predictions.

		Returns:
			dict: Evaluation results with extended metrics.
		"""

		if num_samples is None:

			num_samples = len(dataset)

		else:

			num_samples = min(num_samples, len(dataset))

		results = []
		gt_labels = []
		inference_times = []
		inference_times_single = []
		memory_usages = []
		memory_usages_single = []

		for i in tqdm.tqdm(range(num_samples), desc=f"Evaluating {model_name}", unit="sample"):

			tracemalloc.start()

			data_info = copy.deepcopy(dataset[i])

			try:

				points = data_info['inputs']['points']

			except (KeyError, TypeError):

				print(f"Warning: Unexpected data structure in item {i} | Points")
				continue

			start_time = time.time()

			with torch.no_grad():

				try:

					inference_result = inference_detector(model, points)[0]

				except Exception as e:

					print(f"Inference error for item {i}: {e}")
					tracemalloc.stop()
					continue

			end_time = time.time()

			item_gt_labels = data_info['data_samples'].eval_ann_info['gt_labels_3d'].tolist()

			item_pred_labels = inference_result.pred_instances_3d.labels_3d.cpu().tolist()
			item_pred_scores = inference_result.pred_instances_3d.scores_3d.cpu().tolist()

			item_pred_labels = [label for label, score in zip(item_pred_labels, item_pred_scores) if score >= score_threshold]

			_, peak = tracemalloc.get_traced_memory()
			tracemalloc.stop()

			del data_info

			if item_pred_labels and item_gt_labels:

				results.append(item_pred_labels)
				gt_labels.append(item_gt_labels)

			gc.collect()

			if torch.cuda.is_available():

				torch.cuda.empty_cache()

			memory_usage = peak / 10 ** 6
			memory_usages.append(memory_usage)
			memory_usages_single.append(memory_usage / len(item_gt_labels))

			inference_time = end_time - start_time
			inference_times.append(inference_time)
			inference_times_single.append(inference_time / len(item_gt_labels))

		if not results or not gt_labels:

			print("Not enough results for evaluation")
			return self._get_empty_metrics()

		try:

			y_true = np.concatenate(gt_labels)
			y_pred = np.concatenate(results)

			min_len = min(len(y_true), len(y_pred))
			y_true = y_true[:min_len]
			y_pred = y_pred[:min_len]

			valid_mask = (y_true < len(self.classes)) & (y_pred < len(self.classes)) & (y_true >= 0) & (y_pred >= 0)
			y_true = y_true[valid_mask]
			y_pred = y_pred[valid_mask]

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
			plt.title('Confusion matrix', fontsize=16)
			plt.xlabel('Predicted label', fontsize=12)
			plt.ylabel('True label', fontsize=12)
			plt.xticks(rotation=45, ha='right')
			plt.yticks(rotation=0)
			plt.tight_layout()

			cbar = plt.gca().collections[0].colorbar
			cbar.set_label('Number of samples', labelpad=15)

			plt.savefig(os.path.join('./output', f"{model_name}_confusion_matrix.png"))

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

		empty_metrics = {
			'accuracy': 0,
			'precision_macro': 0,
			'recall_macro': 0,
			'f1_macro': 0,
			'precision_weighted': 0,
			'recall_weighted': 0,
			'f1_weighted': 0,
			'memory_usage_total': 0,
			'memory_usage': 0,
			'memory_usage_std': 0,
			'memory_usage_single': 0,
			'memory_usage_single_std': 0,
			'inference_time_total': 0,
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

	os.makedirs(os.path.dirname(filepath), exist_ok=True)

	with open(filepath, 'w', newline='') as csvfile:

		fieldnames = [k for k in metrics_dict.keys() if k != 'confusion_matrix']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		max_len = max([len(metrics_dict[k]) if isinstance(metrics_dict[k], list) else 1 for k in fieldnames])

		for key in fieldnames:

			if not isinstance(metrics_dict[key], list):

				metrics_dict[key] = [metrics_dict[key]] * max_len

		for i in range(max_len):

			row = {key: metrics_dict[key][i] if i < len(metrics_dict[key]) else None for key in fieldnames}
			writer.writerow(row)

	print(f"Metrics saved to: {filepath}")

if __name__ == "__main__":

	init_default_scope('mmdet3d')
	register_all_modules()

	kitti_root = '/media/pablo/Disco programas/datasets/mmdetection3d/data/kitti/'

	if not os.path.exists(kitti_root):

		print(f"ERROR: The KITTI root directory does not exist: {kitti_root}")

		sys.exit(1)


	mmdet3d_path = '/media/pablo/Disco programas/datasets/mmdetection3d/mmdetection3d'
	config_files = [
		os.path.join(mmdet3d_path, "configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"),
		os.path.join(mmdet3d_path, "configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py"),
	]

	weights_path = '/media/pablo/Disco programas/datasets/mmdetection3d/weights'
	checkpoint_files = [
		os.path.join(weights_path, "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"),
		os.path.join(weights_path, "point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth"),
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
	print(f"Anotations file: {os.path.join(kitti_root, 'kitti_infos_val.pkl')}")

	if not os.path.isfile(os.path.join(kitti_root, 'kitti_infos_val.pkl')):

		print("ERROR: The annotations file does not exist.")

		sys.exit(1)

	try:

		comparer = ModelComparer(config_files, checkpoint_files, device='cuda:0')

		results = comparer.compare_models(dataset_config, num_samples=None, output_dir=output_dir)

		print("\n===== Model comparison results =====")

		for model_name, metrics in results.items():

			print(f"\nResults for {model_name}:")

			for metric, value in metrics.items():

				print(f"\t{metric}: {value}")

	except Exception as e:

		print(f"ERROR: {e}")