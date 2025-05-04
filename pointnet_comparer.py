import os
import sys
import time
import numpy as np
import torch
import csv
import tracemalloc
import gc
import copy
import tqdm
import warnings
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from typing import List, Dict, Tuple, Any, Optional
from easydict import EasyDict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

from pointnet import PointNet
from pointnet_pp import PointNetPlusPlus

class ModelComparer:
	"""
	Class for benchmarking different models.

	This class enables loading multiple point cloud-based 3D models
	and systematically evaluating them, collecting performance metrics such as
	inference time, memory usage, and accuracy.
	"""

	def __init__(self, checkpoint_paths: List[str], dataset_path: str, class_names: List[str], device: str = 'cpu'):
		"""
		Initialize the model benchmark.

		Args:
			checkpoint_paths: List of paths to model checkpoint files.
			dataset_path: Path to the dataset for evaluation.
			class_names: List of class names that the models should detect.
			device: Device for inference ('cuda:0' or 'cpu').

		Raises:
			AssertionError: If the number of config and checkpoint files don't match.
		"""

		self.device = device

		self.models = []
		self.model_names = []

		self.dataset = None
		self.dataloader = None

		self.class_names = class_names

		self.model_memory_footprints = []
		self.benchmark_results = {}

		dataset_config = cfg_from_yaml_file(
			dataset_path,
			EasyDict()
		)

		dataset, dataloader = self._initialize_dataset(dataset_config=dataset_config)

		self.dataset = dataset
		self.dataloader = dataloader

		for checkpoint_path in checkpoint_paths:

			self._load_model(checkpoint_path)

	def _load_model(self, checkpoint_path: str) -> None:
		"""
		Load a model from the specified configuration and checkpoint.

		Args:
			checkpoint_path: Path to the model checkpoint file.
		"""

		tracemalloc.start()

		model_name = os.path.basename(checkpoint_path).split('_')[0]

		self.model_names.append(model_name)

		if model_name == 'pointnet':

			tracemalloc.start()

			new_model = PointNet(num_classes=len(self.class_names))

			checkpoint = torch.load(checkpoint_path, map_location=self.device)
			new_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

			new_model = new_model.to(self.device)
			new_model.eval()

			_, peak_memory = tracemalloc.get_traced_memory()
			tracemalloc.stop()

			self.model_memory_footprints.append(peak_memory / 10 ** 6)

			self.models.append(new_model)

		elif model_name == 'pointnetpp':

			tracemalloc.start()

			new_model = PointNetPlusPlus(num_classes=len(self.class_names))

			checkpoint = torch.load(checkpoint_path, map_location=self.device)
			new_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

			new_model = new_model.to(self.device)
			new_model.eval()

			_, peak_memory = tracemalloc.get_traced_memory()
			tracemalloc.stop()

			self.model_memory_footprints.append(peak_memory / 10 ** 6)

			self.models.append(new_model)

		else:

			raise ValueError(f"Unsupported model type: {model_name}")

	def _initialize_dataset(self, dataset_config: Dict) -> Tuple[Any, Any]:
		"""
		Initialize the dataset for evaluation.

		Args:
			dataset_config: Dataset configuration.

		Returns:
			A tuple (dataset, dataloader) for evaluation.
		"""

		test_dataset, test_loader, _ = build_dataloader(
			dataset_cfg=dataset_config,
			class_names=self.class_names,
			batch_size=1,
			dist=False,
			workers=4,
			training=False
		)

		return test_dataset, test_loader

	@torch.no_grad()
	def evaluate_single_model(self, model_index: int) -> Dict[str, Any]:
		"""
		Evaluate a specific model on its dataset.

		Args:
			model_index: Index of the model to evaluate.

		Returns:
			Dictionary with evaluation metrics.
		"""

		model = self.models[model_index]
		model_name = self.model_names[model_index]

		# Per scene metrics
		inference_times = []
		memory_usages = []

		# Per object metrics
		per_object_inference_times = []
		per_object_memory_usages = []

		pred_results = []
		gt_results = []

		print(f"\n===== Evaluating model: {model_name} =====")

		for batch_idx, data_batch in enumerate(tqdm.tqdm(self.dataloader, desc=f"Evaluating {model_name}")):

			try:

				tracemalloc.start()

				data_batch = copy.deepcopy(data_batch)
				points = data_batch['points']

				points = np.stack([
					points[:, 1],  # X
					points[:, 2],  # Y
					points[:, 3],  # Z
					points[:, 4],  # intensity
				], axis=-1)

				scene_times = []
				item_pred_labels = []
				item_gt_labels = []

				points_tensor = torch.from_numpy(points[:, :3]).unsqueeze(0).float()
				gt_boxes_tensor = torch.from_numpy(data_batch['gt_boxes']).float()[:, :, :7]

				point_masks = points_in_boxes_cpu(points_tensor[0], gt_boxes_tensor[0])

				for i, mask in enumerate(point_masks):

					indices = mask.nonzero(as_tuple=False).squeeze(1)

					if indices.numel() < 32:

						continue

					cluster_points = points[indices.numpy(), :3]

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

					label = int(data_batch['gt_boxes'][0, i, 7]) - 1

					item_pred_labels.append(predicted.item())
					item_gt_labels.append(label)

				_, peak_memory = tracemalloc.get_traced_memory()
				tracemalloc.stop()

				if item_pred_labels:

					assert len(item_pred_labels) == len(item_gt_labels), "Mismatch in predicted and ground truth labels length."

					pred_results.append(np.array(item_pred_labels))
					gt_results.append(np.array(item_gt_labels))

					memory_usage = peak_memory / 10 ** 6

					memory_usages.append(memory_usage)
					inference_times.append(np.sum(scene_times))

					per_object_memory_usages.append(memory_usage / len(item_gt_labels))
					per_object_inference_times.append(np.mean(scene_times))

				del data_batch

				gc.collect()

				if self.device == 'cuda:0' and torch.cuda.is_available():

					torch.cuda.empty_cache()

			except Exception as e:

				print(f"Error processing sample {batch_idx}: {e}")

				traceback.print_exc()

				if tracemalloc.is_tracing():

					tracemalloc.stop()

				continue

		if not pred_results or not gt_results:

			print(f"Model {model_name} evaluation failed. No results or ground truth data found.")

			return self._get_empty_metrics()

		try:

			labels_true = np.concatenate(gt_results)
			labels_pred = np.concatenate(pred_results)

			accuracy = accuracy_score(labels_true, labels_pred)
			precision_by_class = precision_score(labels_true, labels_pred, average=None, zero_division=0, labels=range(len(self.class_names)))
			recall_by_class = recall_score(labels_true, labels_pred, average=None, zero_division=0, labels=range(len(self.class_names)))
			f1_by_class = f1_score(labels_true, labels_pred, average=None, zero_division=0, labels=range(len(self.class_names)))

			precision_macro = precision_score(labels_true, labels_pred, average='macro', zero_division=0)
			recall_macro = recall_score(labels_true, labels_pred, average='macro', zero_division=0)
			f1_macro = f1_score(labels_true, labels_pred, average='macro', zero_division=0)

			precision_weighted = precision_score(labels_true, labels_pred, average='weighted', zero_division=0)
			recall_weighted = recall_score(labels_true, labels_pred, average='weighted', zero_division=0)
			f1_weighted = f1_score(labels_true, labels_pred, average='weighted', zero_division=0)

			cm = confusion_matrix(labels_true, labels_pred, labels=range(len(self.class_names)))

		except Exception as e:

			print(f"Error calculating metrics: {e}")

			traceback.print_exc()

			return self._get_empty_metrics()

		performance_metrics = {
			'accuracy': accuracy,
			'precision_macro': precision_macro,
			'recall_macro': recall_macro,
			'f1_macro': f1_macro,
			'precision_weighted': precision_weighted,
			'recall_weighted': recall_weighted,
			'f1_weighted': f1_weighted,
			'memory_usage_mean_mb': np.mean(memory_usages),
			'memory_usage_std_mb': np.std(memory_usages),
			'per_object_memory_usage_mean_mb': np.mean(per_object_memory_usages),
			'per_object_memory_usage_std_mb': np.std(per_object_memory_usages),
			'inference_time_mean_sec': np.mean(inference_times),
			'inference_time_std_sec': np.std(inference_times),
			'per_object_inference_time_mean_sec': np.mean(per_object_inference_times),
			'per_object_inference_time_std_sec': np.std(per_object_inference_times),
			'num_samples': len(self.dataloader),
			'num_classes': len(self.class_names),
			'model_memory_footprint_mb': self.model_memory_footprints[model_index],
			'confusion_matrix': cm
		}

		for i, cls in enumerate(self.class_names):

			performance_metrics[f'{cls.lower()}_precision'] = precision_by_class[i]
			performance_metrics[f'{cls.lower()}_recall'] = recall_by_class[i]
			performance_metrics[f'{cls.lower()}_f1'] = f1_by_class[i]

		self._print_evaluation_summary(model_name, performance_metrics)

		return performance_metrics

	def _get_empty_metrics(self):
		"""
		Returns:
			Empty metrics dictionary when evaluation fails.
		"""

		empty_metrics = {
			'accuracy': 0,
			'precision_macro': 0,
			'recall_macro': 0,
			'f1_macro': 0,
			'precision_weighted': 0,
			'recall_weighted': 0,
			'f1_weighted': 0,
			'memory_usage_mean_mb': 0,
			'memory_usage_std_mb': 0,
			'per_object_memory_usage_mean_mb': 0,
			'per_object_memory_usage_std_mb': 0,
			'inference_time_mean_sec': 0,
			'inference_time_std_sec': 0,
			'per_object_inference_time_mean_sec': 0,
			'per_object_inference_time_std_sec': 0,
			'num_samples': 0,
			'num_classes': 0,
			'model_memory_footprint_mb': 0,
			'confusion_matrix': None
		}

		for cls in self.class_names:

			empty_metrics[f'{cls.lower()}_precision'] = 0
			empty_metrics[f'{cls.lower()}_recall'] = 0
			empty_metrics[f'{cls.lower()}_f1'] = 0

		return empty_metrics

	def _print_evaluation_summary(self, model_name: str, metrics: Dict[str, Any]) -> None:
		"""
		Print a summary of the model evaluation.

		Args:
			model_name: Name of the evaluated model.
			metrics: Model evaluation metrics.
		"""

		print("\n===== Model Evaluation Summary =====")
		print(f"Model: {model_name}")
		print(f"Memory usage (MB): {metrics['memory_usage_mean_mb']:.4f} ± {metrics['memory_usage_std_mb']:.4f}")
		print(f"Inference time (s): {metrics['inference_time_mean_sec']:.4f} ± {metrics['inference_time_std_sec']:.4f}")
		print(f"Memory usage per object (MB): {metrics['per_object_memory_usage_mean_mb']:.4f} ± {metrics['per_object_memory_usage_std_mb']:.4f}")
		print(f"Inference time per object (s): {metrics['per_object_inference_time_mean_sec']:.4f} ± {metrics['per_object_inference_time_std_sec']:.4f}")
		print(f"Model memory footprint (MB): {metrics['model_memory_footprint_mb']:.4f}")
		print(f"Number of samples: {metrics['num_samples']}")
		print(f"Number of classes: {metrics['num_classes']}")
		print(f"Accuracy: {metrics['accuracy']:.4f}")
		print(f"Precision (macro): {metrics['precision_macro']:.4f}")
		print(f"Recall (macro): {metrics['recall_macro']:.4f}")
		print(f"F1 Score (macro): {metrics['f1_macro']:.4f}")
		print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
		print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
		print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
		print(f"Detailed metrics by class:")
		for cls in self.class_names:
			print(f"{cls}:")
			print(f"  Precision: {metrics[f'{cls}_precision']:.4f}")
			print(f"  Recall: {metrics[f'{cls}_recall']:.4f}")
			print(f"  F1 Score: {metrics[f'{cls}_f1']:.4f}")
		print("\n===================================")

	def run_benchmark(self, output_dir: Optional[str] = None, images_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
		"""
		Run the benchmark evaluation for all loaded models.

		Args:
			output_dir: Directory to save results. If None, nothing is saved.
			images_dir: Directory to save images. If None, nothing is saved.

		Returns:
			Dictionary with evaluation results for all models.
		"""

		for model_idx, model_name in enumerate(self.model_names):

			print(f"\n===== Evaluating model {model_idx + 1}/{len(self.models)}: {model_name} =====")

			self.benchmark_results[model_name] = self.evaluate_single_model(model_idx)

			if output_dir:

				results_path = os.path.join(output_dir, f"{model_name}_real_classification_results.csv")
				self._save_metrics_to_csv(self.benchmark_results[model_name], results_path)

			if images_dir:

				cm = self.benchmark_results[model_name]['confusion_matrix']
				class_names = self.class_names

				self._generate_confusion_matrix_plot(cm, class_names, model_name, images_dir)

		return self.benchmark_results

	def _generate_confusion_matrix_plot(self, cm: np.ndarray, class_names: List[str], model_name: str, output_dir: str) -> None:
		"""
		Generate and save a confusion matrix plot.

		Args:
			cm: Confusion matrix.
			class_names: List of class names.
			model_name: Name of the model.
			output_dir: Directory to save the plot.
		"""

		if cm is None:

			print(f"Confusion matrix is None for model {model_name}. Skipping plot generation.")

			return

		plt.figure(figsize=(10, 8))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, linewidths=0.5, linecolor='gray', cbar=True, square=True)

		plt.title('Confusion Matrix', fontsize=16)
		plt.xlabel('Predicted Label', fontsize=12)
		plt.ylabel('True Label', fontsize=12)
		plt.xticks(rotation=45, ha='right')
		plt.yticks(rotation=0)
		plt.tight_layout()

		cbar = plt.gca().collections[0].colorbar
		cbar.set_label('Number of samples', labelpad=15)

		plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
		plt.close()

	def _save_metrics_to_csv(self, metrics_dict: Dict[str, Any], filepath: str) -> None:
		"""
		Save metrics to a CSV file.

		Args:
			metrics_dict: Dictionary with metrics to save.
			filepath: Path where to save the CSV file.
		"""

		cm = metrics_dict.pop('confusion_matrix', None)

		os.makedirs(os.path.dirname(filepath), exist_ok=True)

		with open(filepath, 'a', newline='') as csvfile:

			fieldnames = [key for key in metrics_dict.keys() if key not in ['confusion_matrix']]
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			if os.path.getsize(filepath) == 0:

				writer.writeheader()

			writer.writerow(metrics_dict)

		print(f"Metrics saved to {filepath}")

		metrics_dict['confusion_matrix'] = cm

def main():
	"""
	Main function to run the 3D Pointnet and Pointnet++ benchmark.
	"""

	target_classes = ['Car', 'Pedestrian', 'Cyclist']
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	output_dir = './output'
	images_dir = './images'

	openpcdet_path = '/media/pablo/Disco programas/datasets/openpcdet/OpenPCDet'
	dataset_path = os.path.join(openpcdet_path, 'tools/cfgs/dataset_configs/kitti_dataset.yaml')

	config_path = '/home/pablo/Desktop/pointnet/output/'
	checkpoint_paths = [
		os.path.join(config_path, "pointnet_best.pth"),
		os.path.join(config_path, "pointnetpp_best.pth"),
	]

	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(images_dir, exist_ok=True)

	try:

		print("===== Model Benchmark =====")
		print(f"Device: {device}")
		print(f"Output directory: {output_dir}")
		print(f"Images directory: {images_dir}")
		print(f"Target classes: {target_classes}")
		print(f"Config paths: {None}")
		print(f"Checkpoint paths: {checkpoint_paths}")
		print("============================")

		benchmark = ModelComparer(
			checkpoint_paths=checkpoint_paths,
			dataset_path=dataset_path,
			class_names=target_classes,
			device=device
		)

		benchmark.run_benchmark(output_dir=output_dir, images_dir=images_dir)
		print("\n===== Benchmark complete =====")
		print(f"Results saved to: {output_dir}")

		return 0

	except Exception as e:

		print(f"ERROR: {e}")

		traceback.print_exc()

		return 1


if __name__ == "__main__":

	sys.exit(main())