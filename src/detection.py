"""
Running this script trains a vision model for object detection.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics import YOLO
import time
from src.utils.get_args import get_args, update_yaml
from argparse import ArgumentParser
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# file to save performance measurements and the columns to put them into.
output_file = "output/results.csv"
headers = ['data', 'epochs', 'hot_bat_precision', 'hot_bat_recall', 'hot_bat_mAP50', 'hot_bat_mAP50_95']

def estimate_model(data_yaml, starting_model="yolov8n.pt", epochs=100, workers=1, show_time=True, **args):
    """
    `estimate_model` is the function that initiates training the vision model for object detection.

    Args:
        data_yaml (str): Path to the YAML file for the dataset. It should specify at least a `train` slice, and generally `test` and `val` slices too.
        starting_model (str): Path to the PyTorch weights of the model to be trained. If a YOLO base model (e.g., `yolov8n`) is specified then it will be downloaded for training unless it is already present.
        epochs (int): Maximum number of epochs to train the detection model. Training may stop before reaching this limit if accuracy is not improving.
        workers (int): Number of parallel threads that will be used for loading training data. We've run out of memory if `workers` is set greater than one.
        show_time (bool): True/False flag that determines whether to print the elapsed time to the console at the end of training.

    Returns:
        result: The result of the training process, including the model's performance on the validation set
    """
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    start_time = time.time()

    model = YOLO(starting_model).to(device)

    model_train_args = {"data": data_yaml, "epochs": epochs, "workers": workers}
    for key in [
        "name",
        "imgsz",
        "batch",
        "patience",
        "optimizer",
        "seed",
        "deterministic",
        "save_period"
    ]:
        if key in args:
            model_train_args[key] = args[key]

    if "fitted_models_dir" in args:
        model_train_args["project"] = args["fitted_models_dir"]

    result = model.train(**model_train_args)

    if show_time:
        print(f"--- {round(time.time() - start_time, 2)} seconds ---")
    
    new_row = [data_yaml, epochs]
    new_row.extend(result.class_result(1))

    return result


def model_results(
        results,
        data_yaml: str,
        copies: int,
        starting_model: str = "yolov8n.pt",
        epochs: int = np.nan,
        **kwargs,
    ) -> pd.DataFrame:
    """
    Extracts performance metrics from a YOLO model validation run and saves them to a CSV.
    Outputs the confusion matrix and related metrics of a trained YOLO model on test images.

    Mainly used to test transform pipelines and n_copies combinations for test sets. Saves
    confusion matrix data to a csv. Future purpose to test generalizability.

    Author: Eric Sun
    """
    dataset_settings = get_args(data_yaml)

    training_values = [dataset_settings["train"], dataset_settings["val"], starting_model, copies, epochs]
    training_columns = [
        "training_data",
        "validation_data",
        "starting_model",
        "img_copies",
        "epochs",
    ]

    cold_metrics = results.box.class_result(0)
    hot_metrics = results.box.class_result(1)

    metric_labels = ["precision", "recall", "ap50", "ap"]

    cold_bat_labels = [f"{metric}_cold_bat" for metric in metric_labels]
    hot_bat_labels = [f"{metric}_hot_bat" for metric in metric_labels]

    training_values.extend(cold_metrics + hot_metrics)
    training_columns.extend(cold_bat_labels + hot_bat_labels)

    truth = ["true_cold", "true_hot", "true_background"]
    predicted = ["pred_cold", "pred_hot", "pred_background"]

    matrix_values = results.confusion_matrix.matrix.flatten().tolist()
    matrix_values = [int(x) for x in matrix_values]
    matrix_names = [f"{p}_{t}" for p in predicted for t in truth]

    training_values.extend(matrix_values)
    training_columns.extend(matrix_names)

    return pd.DataFrame([training_values], columns=training_columns)


if __name__ == "__main__":
    parser = ArgumentParser(description="Use the YAML config file to provide settings for the model.")
    parser.add_argument("-c", "--config", dest="config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    settings = get_args(args.config_path)

    if "detection" in settings:
        settings["detection"]["name"] = Path(args.config_path).stem
        estimation_result = estimate_model(data_yaml=settings["data_yaml"], **settings["detection"])
        settings_dict = settings["detection"]
    else:
        settings["name"] = Path(args.config_path).stem
        estimation_result = estimate_model(**settings)
        settings_dict = settings

    trained_model_path = str(estimation_result.save_dir / "weights/best.pt")
    best_confidence = float(
        estimation_result.curves_results[1][0][
            np.argmax(estimation_result.curves_results[1][1][0, :])
        ]
    )

    dynamic_args = {
        "tracking.model_file": trained_model_path,
        "tracking.detection_confidence": best_confidence,
    }
    update_yaml(args.config_path, **dynamic_args)

    if "output_file" in settings_dict:
        result_df = model_results(
            results=estimation_result,
            data_yaml=settings["data_yaml"],
            copies=settings["n_copies"],
            **settings_dict,
        )
        result_df.to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(settings_dict["output_file"]),
            index=False
        )
