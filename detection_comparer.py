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

from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from typing import List, Dict, Tuple, Any, Optional

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
		model_config = self.model_configs[model_index]
		model_name = self.model_names[model_index]
		dataset = self.datasets[model_index]
		dataloader = self.dataloaders[model_index]

		# Per scene metrics
		inference_times = []
		memory_usages = []
		gpu_memory_usages = []

		# Per object metrics
		per_object_inference_times = []
		per_object_memory_usages = []
		per_object_gpu_memory_usages = []

		detection_annotations = []

		print(f"\n===== Evaluating model: {model_name} =====")

		for batch_idx, data_batch in enumerate(tqdm.tqdm(dataloader, desc=f"Evaluating {model_name}")):

			try:

				tracemalloc.start()

				data_batch = copy.deepcopy(data_batch)

				if self.device.startswith('cuda') and torch.cuda.is_available():

					torch.cuda.reset_peak_memory_stats()
					torch.cuda.empty_cache()

				if self.device == 'cuda:0' and torch.cuda.is_available():

					load_data_to_gpu(data_batch)

				start_time = time.time()
				prediction_dicts, _ = model.forward(data_batch)
				end_time = time.time()

				_, peak_memory = tracemalloc.get_traced_memory()
				tracemalloc.stop()

				if self.device.startswith('cuda') and torch.cuda.is_available():

					gpu_mem = torch.cuda.max_memory_allocated(device=self.device)
					gpu_mem_mb = gpu_mem / (1024 * 1024)
					gpu_memory_usages.append(gpu_mem_mb)

				else:

					gpu_mem_mb = 0
					gpu_memory_usages.append(gpu_mem_mb)

				memory_usage = peak_memory / (1024 * 1024)
				memory_usages.append(memory_usage)

				inference_time = end_time - start_time
				inference_times.append(inference_time)

				num_objects_in_batch = len(data_batch['gt_boxes'][0])

				per_object_memory_usages.append(memory_usage / num_objects_in_batch)
				per_object_inference_times.append(inference_time / num_objects_in_batch)
				per_object_gpu_memory_usages.append(gpu_mem_mb / num_objects_in_batch)

				annotations = dataset.generate_prediction_dicts(
					batch_dict=data_batch,
					pred_dicts=prediction_dicts,
					class_names=dataset.class_names
				)
				detection_annotations.extend(annotations)

				del data_batch, prediction_dicts

				gc.collect()

				if self.device == 'cuda:0' and torch.cuda.is_available():

					torch.cuda.empty_cache()

			except Exception as e:

				print(f"Error processing sample {batch_idx}: {e}")

				traceback.print_exc()

				if tracemalloc.is_tracing():

					tracemalloc.stop()

				continue

		result_str, detailed_results = dataset.evaluation(
			detection_annotations,
			dataset.class_names,
			eval_metric=model_config.MODEL.POST_PROCESSING.EVAL_METRIC
		)

		# Use default str representation for detailed results terminal logging
		print(result_str)

		performance_metrics = {
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
			'confusion_matrix': None,
			'detection_metrics': detailed_results
		}

		self._print_evaluation_summary(model_name, performance_metrics)

		return performance_metrics

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

				results_path = os.path.join(output_dir, f"{model_name}_real_detection_results.csv")
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

		os.makedirs(os.path.dirname(filepath), exist_ok=True)

		cm = metrics_dict.pop('confusion_matrix', None)

		detection_metrics = metrics_dict.pop('detection_metrics', {})

		flattened_metrics = {}

		for key, value in metrics_dict.items():

			if isinstance(value, (int, float, str, bool)) or value is None:

				flattened_metrics[key] = value

			else:

				flattened_metrics[key] = str(value)

		for metric_name, metric_value in detection_metrics.items():

			if isinstance(metric_value, dict):

				for subkey, subvalue in metric_value.items():

					flattened_metrics[f"detection_{metric_name}_{subkey}"] = subvalue

			else:

				flattened_metrics[f"detection_{metric_name}"] = metric_value

		with open(filepath, 'w', newline='') as csvfile:

			fieldnames = list(flattened_metrics.keys())

			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			writer.writeheader()
			writer.writerow(flattened_metrics)

		print(f"Metrics saved to: {filepath}")

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