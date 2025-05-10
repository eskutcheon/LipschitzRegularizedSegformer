from typing import Callable, Union, Literal, Any, List, Dict, Type, Tuple
import torchmetrics as TM
import torch
import torch.nn.functional as F


# consider changing the averaging type if we want more info about the performance by class (default = 'macro')
    # can be extended at any time with the way I'm considering sampling from them
#? NOTE: refactored to a dictionary to store keyword arguments for each metric more clearly
supported_metrics: Dict[str, Dict[str, Any]] = {
    'iou': {"maximize": True, "metric": TM.JaccardIndex, "kwargs": {"average": 'macro'}},
    'accuracy': {"maximize": True, "metric": TM.Accuracy, "kwargs": {"average": 'macro'}},
    'precision': {"maximize": True, "metric": TM.Precision, "kwargs": {"average": 'macro'}},
    'recall': {"maximize": True, "metric": TM.Recall, "kwargs": {"average": 'macro'}},
    'f1': {"maximize": True, "metric": TM.F1Score, "kwargs": {"average": 'macro'}},
    "confusion": {"maximize": False, "metric": TM.ConfusionMatrix, "kwargs": {"normalize": "true"}},
    #? NOTE: removing for now - only works on FloatTensor
    #"calibration": {"maximize": False, "metric": TM.CalibrationError, "kwargs": {"n_bins": 15, "norm": "l1"}},
    "phi": {"maximize": True, "metric": TM.MatthewsCorrCoef, "kwargs": {}}
}

def get_MetricTracker(test_metrics: List[str], num_classes: int, device: str) -> TM.MetricTracker:
    ''' constructs a TM.MetricTracker from requested metrics or implemented TM.Metrics objects and returns the corresponding objects
        Args:
            test_metrics: list of metrics to add to the tracker - defined with CLI
            tracked_metrics: dict of metrics implemented elsewhere. May be an empty dict
            max_flags: boolean array of whether or not bigger values are better
        Return:
            tracker comprised of Metric objects: TM.MetricTracker(TM.MetricCollection(TM.Metric))
    '''
    max_flags: List[bool] = []
    tracked_metrics: dict = {}
    metric_kwargs = {"task": 'multiclass', "num_classes": num_classes, "validate_args": False}
    for metric in test_metrics:
        if metric not in supported_metrics:
            print(f"WARNING: {metric} is not supported. Supported metrics are: {list(supported_metrics.keys())}")
            continue
        metric_info = supported_metrics[metric]
        try:
            # combine default metric arguments with any additional kwargs in `supported_metrics`
            kwargs = {**metric_kwargs, **metric_info["kwargs"]}
            tracked_metrics[metric] = metric_info["metric"](**kwargs).to(device)
            max_flags.append(metric_info["maximize"])
        except Exception as e:
            print(f"Error initializing metric '{metric}': {e}")
            print("Skipping this metric...")
    # ? NOTE: implicit optimization - compute_groups parameter of TM.MetricCollection set True by default
    return TM.MetricTracker(TM.MetricCollection(tracked_metrics), maximize=max_flags)