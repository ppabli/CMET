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
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from pcdet.utils import common_utils

class ModelComparer:
	"""
	Class for benchmarking different models.

	This class enables loading multiple point cloud-based 3D models
	and systematically evaluating them, collecting performance metrics such as
	inference time, memory usage, and accuracy.
	"""

	def __init__(self, config_paths: List[str], checkpoint_paths: List[str], class_names: List[str], device: str = 'cpu'):
		"""
		Initialize the model benchmark.

		Args:
			config_paths: List of paths to model configuration files.
			checkpoint_paths: List of paths to model checkpoint files.
			class_names: List of class names that the models should detect.
			device: Device for inference ('cuda:0' or 'cpu').

		Raises:
			AssertionError: If the number of config and checkpoint files don't match.
		"""

		assert len(config_paths) == len(checkpoint_paths), "Number of config and checkpoint files must match"

		self.device = device
		self.models = []
		self.model_configs = []
		self.model_names = []
		self.datasets = []
		self.dataloaders = []
		self.class_names = class_names
		self.logger = common_utils.create_logger()
		self.model_memory_footprints = []
		self.model_gpu_memory_footprints = []
		self.benchmark_results = {}

		for config_path, checkpoint_path in zip(config_paths, checkpoint_paths):

			self._load_model(config_path, checkpoint_path)

	def _load_model(self, config_path: str, checkpoint_path: str) -> None:
		"""
		Load a model from the specified configuration and checkpoint.

		Args:
			config_path: Path to the model configuration file.
			checkpoint_path: Path to the model checkpoint file.
		"""

		if self.device.startswith('cuda') and torch.cuda.is_available():

			torch.cuda.empty_cache()
			torch.cuda.reset_peak_memory_stats()

		tracemalloc.start()

		model_config = cfg_from_yaml_file(config_path, EasyDict())

		dataset, dataloader = self._initialize_dataset(dataset_config=model_config.DATA_CONFIG)

		model_name = os.path.basename(config_path).split('.')[0]

		model = build_network(
			model_cfg=model_config.MODEL,
			num_class=len(model_config.CLASS_NAMES),
			dataset=dataset
		)

		model.load_params_from_file(filename=checkpoint_path, logger=self.logger)

		model.to(self.device)
		model.eval()

		if self.device.startswith('cuda') and torch.cuda.is_available():

			model_gpu_mem = torch.cuda.max_memory_allocated(device=self.device) / (1024 * 1024)

		else:

			model_gpu_mem = 0

		self.model_gpu_memory_footprints.append(model_gpu_mem)

		_, peak_memory = tracemalloc.get_traced_memory()
		tracemalloc.stop()

		model_memory_footprint = peak_memory / (1024 * 1024)
		self.model_memory_footprints.append(model_memory_footprint)

		self.models.append(model)
		self.model_configs.append(model_config)
		self.model_names.append(model_name)
		self.datasets.append(dataset)
		self.dataloaders.append(dataloader)

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
		dataloader = self.dataloaders[model_index]

		# Per scene metrics
		inference_times = []
		memory_usages = []
		gpu_memory_usages = []

		# Per object metrics
		per_object_inference_times = []
		per_object_memory_usages = []
		per_object_gpu_memory_usages = []

		pred_results = []
		gt_results = []

		print(f"\n===== Evaluating model: {model_name} =====")

		for batch_idx, data_batch in enumerate(tqdm.tqdm(dataloader, desc=f"Evaluating {model_name}")):

			try:

				tracemalloc.start()

				if self.device.startswith('cuda') and torch.cuda.is_available():

					torch.cuda.reset_peak_memory_stats()
					torch.cuda.empty_cache()

				data_batch = copy.deepcopy(data_batch)

				gt_boxes = data_batch['gt_boxes']
				gt_labels = [int(data_batch['gt_boxes'][0, i, 7]) - 1 for i in range(data_batch['gt_boxes'].shape[1])]

				points = data_batch['points']

				points = np.stack([
					points[:, 1],  # X
					points[:, 2],  # Y
					points[:, 3],  # Z
					points[:, 4],  # intensity
				], axis=-1)

				points_tensor = torch.from_numpy(points[:, :3]).unsqueeze(0)
				gt_boxes_tensor = torch.from_numpy(gt_boxes)[:, :, :7]

				point_masks = points_in_boxes_cpu(points_tensor[0], gt_boxes_tensor[0])

				scene_times = []
				item_pred_labels = []
				item_gt_labels = []

				for i, mask in enumerate(point_masks):

					indices = mask.nonzero(as_tuple=False).squeeze(1)

					if indices.numel() < 32:

						continue

					object_batch = copy.deepcopy(data_batch)

					object_batch['points'] = data_batch['points'][indices]

					processed_batch = self._preprocess_points_for_model(model_index, object_batch)

					if self.device == 'cuda:0' and torch.cuda.is_available():

						load_data_to_gpu(processed_batch)

					start_time = time.time()
					prediction_dicts, _ = model.forward(processed_batch)
					end_time = time.time()

					pred_labels = prediction_dicts[0]['pred_labels'].cpu().numpy()

					if len(pred_labels) == 0:

						continue

					pred_label = pred_labels[0]

					item_pred_labels.append(pred_label - 1)
					item_gt_labels.append(gt_labels[i])

					scene_times.append(end_time - start_time)

				_, peak_memory = tracemalloc.get_traced_memory()
				tracemalloc.stop()

				if self.device.startswith('cuda') and torch.cuda.is_available():

					gpu_mem = torch.cuda.max_memory_allocated(device=self.device)
					gpu_mem_mb = gpu_mem / (1024 * 1024)
					gpu_memory_usages.append(gpu_mem_mb)

				else:

					gpu_mem_mb = 0
					gpu_memory_usages.append(gpu_mem_mb)

				if item_pred_labels:

					assert len(item_pred_labels) == len(item_gt_labels), "Mismatch in predicted and ground truth labels length."

					pred_results.append(np.array(item_pred_labels))
					gt_results.append(np.array(item_gt_labels))

					memory_usage = peak_memory / 10 ** 6

					memory_usages.append(memory_usage)
					inference_times.append(np.sum(scene_times))

					per_object_memory_usages.append(memory_usage / len(item_gt_labels))
					per_object_inference_times.append(np.mean(scene_times))
					per_object_gpu_memory_usages.append(gpu_mem_mb / len(item_gt_labels))

				del data_batch

				gc.collect()

				if self.device == 'cuda:0' and torch.cuda.is_available():

					torch.cuda.empty_cache()

			except Exception as e:

				print(f"Error processing sample {batch_idx}: {e}")

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
			'gpu_memory_usage_mean_mb': np.mean(gpu_memory_usages),
			'gpu_memory_usage_std_mb': np.std(gpu_memory_usages),
			'per_object_memory_usage_mean_mb': np.mean(per_object_memory_usages),
			'per_object_memory_usage_std_mb': np.std(per_object_memory_usages),
			'per_object_gpu_memory_usage_mean_mb': np.mean(per_object_gpu_memory_usages),
			'per_object_gpu_memory_usage_std_mb': np.std(per_object_gpu_memory_usages),
			'inference_time_mean_sec': np.mean(inference_times),
			'inference_time_std_sec': np.std(inference_times),
			'per_object_inference_time_mean_sec': np.mean(per_object_inference_times),
			'per_object_inference_time_std_sec': np.std(per_object_inference_times),
			'num_samples': len(dataloader),
			'num_classes': len(self.class_names),
			'model_memory_footprint_mb': self.model_memory_footprints[model_index],
			'model_gpu_memory_footprint_mb': self.model_gpu_memory_footprints[model_index],
			'confusion_matrix': cm
		}

		for i, cls in enumerate(self.class_names):

			performance_metrics[f'{cls}_precision'] = precision_by_class[i]
			performance_metrics[f'{cls}_recall'] = recall_by_class[i]
			performance_metrics[f'{cls}_f1'] = f1_by_class[i]

		self._print_evaluation_summary(model_name, performance_metrics)

		return performance_metrics

	def _get_empty_metrics(self) -> Dict[str, Any]:
		"""
		Get an empty metrics dictionary.
		Used when evaluation fails.

		Args:
			None

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
			'gpu_memory_usage_mean_mb': 0,
			'gpu_memory_usage_std_mb': 0,
			'per_object_memory_usage_mean_mb': 0,
			'per_object_memory_usage_std_mb': 0,
			'per_object_gpu_memory_usage_mean_mb': 0,
			'per_object_gpu_memory_usage_std_mb': 0,
			'inference_time_mean_sec': 0,
			'inference_time_std_sec': 0,
			'per_object_inference_time_mean_sec': 0,
			'per_object_inference_time_std_sec': 0,
			'num_samples': 0,
			'num_classes': 0,
			'model_memory_footprint_mb': 0,
			'model_gpu_memory_footprint_mb': 0,
			'confusion_matrix': None
		}

		for cls in self.class_names:

			empty_metrics[f'{cls}_precision'] = 0
			empty_metrics[f'{cls}_recall'] = 0
			empty_metrics[f'{cls}_f1'] = 0

		return empty_metrics

	def _preprocess_points_for_model(self, model_index: int, data_batch: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Preprocess the input data for the specified model.
		Args:
			model_index: Index of the model to preprocess data for.
			data_batch: Input data batch.
		Returns:
			Preprocessed data batch for the model.
		"""

		model_name = self.model_names[model_index]
		dataset = self.datasets[model_index]

		batch_dict = {
			'batch_size': 1,
		}

		processed_data = dataset.prepare_data({'points': data_batch['points'][:, 1:]})
		batch_dict.update(processed_data)

		if model_name == 'pointpillar':

			batch_dict['voxel_coords'] = np.concatenate((np.zeros((batch_dict['voxel_coords'].shape[0], 1), dtype=np.int32), batch_dict['voxel_coords']), axis=1)

		if model_name == 'pointrcnn':

			batch_dict['points'] = np.concatenate((np.zeros((batch_dict['points'].shape[0], 1), dtype=np.int32), batch_dict['points']), axis=1)

		return batch_dict

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
		print(f"GPU memory usage (MB): {metrics['gpu_memory_usage_mean_mb']:.4f} ± {metrics['gpu_memory_usage_std_mb']:.4f}")
		print(f"Inference time (s): {metrics['inference_time_mean_sec']:.4f} ± {metrics['inference_time_std_sec']:.4f}")
		print(f"Memory usage per object (MB): {metrics['per_object_memory_usage_mean_mb']:.4f} ± {metrics['per_object_memory_usage_std_mb']:.4f}")
		print(f"GPU memory usage per object (MB): {metrics['per_object_gpu_memory_usage_mean_mb']:.4f} ± {metrics['per_object_gpu_memory_usage_std_mb']:.4f}")
		print(f"Inference time per object (s): {metrics['per_object_inference_time_mean_sec']:.4f} ± {metrics['per_object_inference_time_std_sec']:.4f}")
		print(f"Model memory footprint (MB): {metrics['model_memory_footprint_mb']:.4f}")
		print(f"Model GPU memory footprint (MB): {metrics['model_gpu_memory_footprint_mb']:.4f}")
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

				results_path = os.path.join(output_dir, f"{model_name}_classification_results.csv")
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

		Returns:
			None
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

		Returns:
			None
		"""

		cm = metrics_dict.pop('confusion_matrix', None)

		os.makedirs(os.path.dirname(filepath), exist_ok=True)

		with open(filepath, 'a', newline='') as csvfile:

			fieldnames = list(metrics_dict.keys())
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			if os.path.getsize(filepath) == 0:

				writer.writeheader()

			writer.writerow(metrics_dict)

		print(f"Metrics saved to {filepath}")

		metrics_dict['confusion_matrix'] = cm

def main():
	"""
	Main function to run the 3D detection model benchmark.
	"""

	target_classes = ['Car', 'Pedestrian', 'Cyclist']
	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	output_dir = './output'

	openpcdet_path = '.'
	config_paths = [
		os.path.join(openpcdet_path, "tools/cfgs/kitti_models/pointpillar.yaml"),
		os.path.join(openpcdet_path, "tools/cfgs/kitti_models/pointrcnn.yaml"),
	]

	checkpoint_paths = [
		os.path.join(openpcdet_path, "weights/pointpillar_7728.pth"),
		os.path.join(openpcdet_path, "weights/pointrcnn_7870.pth"),
	]

	os.makedirs(output_dir, exist_ok=True)

	try:

		print("===== Model Benchmark =====")
		print(f"Device: {device}")
		print(f"Output directory: {output_dir}")
		print(f"Images directory: {None}")
		print(f"Target classes: {target_classes}")
		print(f"Config paths: {None}")
		print(f"Checkpoint paths: {checkpoint_paths}")
		print("============================")

		benchmark = ModelComparer(
			config_paths=config_paths,
			checkpoint_paths=checkpoint_paths,
			class_names=target_classes,
			device=device
		)

		benchmark.run_benchmark(output_dir=output_dir)
		print("\n===== Benchmark complete =====")
		print(f"Results saved to: {output_dir}")

		return 0

	except Exception as e:

		print(f"ERROR: {e}")

		traceback.print_exc()

		return 1


if __name__ == "__main__":

	sys.exit(main())