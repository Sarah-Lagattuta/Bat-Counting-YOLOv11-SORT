"""
This script uses a trained computer vision model to detect warm bats in a video, and uses the SORT algorithm to track bats that appear across mutiple frames of video. Finally, it generates a count of the unique bats that were tracked in the video.

Usage:
    python -m tracking --config=path/to/config.yaml
"""

import os
from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import subprocess

from sort.sort import Sort  # Import the SORT tracker
import time
from src.utils.get_args import get_args
from argparse import ArgumentParser, BooleanOptionalAction
from src.bg_subtract_new import background_subtract
import tempfile
from pathlib import Path

def annotate_video(video, destination, annotations, region):
    """
    Annotate a video with bounding boxes and labels, and save it to the destination path.

    Args:
        video (str): Path to the video file to annotate
        destination (str): Path to save the annotated video
        annotations (list): List of lists of bounding boxes and labels for each frame
        region (list): Coordinates of the region to track bats in [x1, y1, x2, y2]
    
    Returns:
        None
    """

    # Open the video file and get its properties
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a VideoWriter object to write the annotated video
    os.makedirs(Path(destination).parent, exist_ok=True)
    out = cv2.VideoWriter(destination, fourcc, fps, (frame_width, frame_height))
    
    # establish the region where bats are being tracked
    region = np.array(region)
    region[[0, 2]] = region[[0, 2]] * frame_width
    region[[1, 3]] = region[[1, 3]] * frame_height

    i = 0
    # Iterate through the frames of the input video, adding annotations and writing to the output video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        # get the annotations for the current frame
        try:
            tracked_objects = annotations.pop(0)
        except IndexError:
            print(f"While annotating {destination}, ran out of annotations before frames.")
            break

        # Add annotations to the frame
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f"ID: {track_id}", (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # add an annotation for the tracked region
        x1, y1, x2, y2 = region.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "Tracked Region", (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


        # write the resulting frame to the output video(s)
        out.write(frame)
    
    # Close the video files
    cap.release()
    out.release()


def centered_in_region(box, region, shape):
    """
    Check if a bounding box is centered within a region.
    
    Args:
        box (list): Bounding box coordinates [x1, y1, x2, y2]
        region (list): Region coordinates [x1, y1, x2, y2]
    
    Returns:
        bool: True if the box is centered within the region, False otherwise
    """
    box_center = [(box[0] + box[2]) / shape[1] / 2.0, (box[1] + box[3]) / shape[0] / 2.0]
    
    return (region[0] < box_center[0] < region[2] and
            region[1] < box_center[1] < region[3])


def check_videos(videos):

    # create a list to store video information
    verified_videos = []

    # Get details of the video from the config file (may be a list or a dict)
    for k in range(len(videos)):

        # default settings
        video_settings = {
            "tracking_region": [0, 0, 1, 1],
            "amplification": 1.0
        }

        if type(videos[k]) is str:
            video_path = videos[k]
            video_settings["path"] = video_path
        elif type(videos[k]) is dict and "path" in videos[k].keys():
            video_dict = videos[k]
            video_path = video_dict["path"]
            video_settings.update(video_dict)
            """# Get the region to track bats in
            if region is not None:
                if type(region) is list and len(region) == 4 and all([x >= 0 and x <= 1 for x in region]):
                    region = region
                else:
                    raise ValueError("Region must be a list of four numbers between 0 and 1.")"
            """
        else:
            raise ValueError(f"Video {videos[k]} details in the config file aren't valid (should be a dict with a path entry, or a string ).")

        # Check if the video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
        
        verified_videos.append(video_settings)

    return verified_videos
                
def convert_to_avi(input_path, temp_dir):
        base = Path(input_path).stem
        output_path = os.path.join(temp_dir, f"{base}_converted.avi")

        cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", "mjpeg",
                "-q:v", "2",
                output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path

def track_bats(model_file: str, video_files: str, detection_confidence: float = 0.2, print_time: bool = True, sort = None, write_annotated_frames="", **kwargs) -> None:
    """
    Track warm bats in videos using the SORT algorithm.

    Args:
        model_file (str): Path to the YOLOv5 model file
        video_files (list): List of video files to track bats in
        detection_confidence (float): Confidence threshold for detection
        print_time (bool): Print the time taken to process each video
        sort (dict): Arguments for the SORT tracker
        write_annotated_frames (str): Path to save annotated videos
        **kwargs: Additional arguments
    
    Returns:
        None
    """
    # provide default arguments to SORT in case they aren't passed to the function
    sort_args = {'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3}
    if sort is not None:
        sort_args.update(sort)
    
    # Initialize SORT tracker
    tracker = Sort(**sort_args)
    model = YOLO(model_file)

    print(f"confidence for detection: {detection_confidence}")
    start_time = time.time()
    result_df = None

    # Iterate through videos
    for k in range(len(video_files)):

        video = video_files[k]["path"]

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for tracking: {video}")

        print(f"\n--- DEBUG VIDEO INFO ---")
        print(f"Tracking video: {video}")
        print(f"Opened: {cap.isOpened()}")
        print(f"Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(f"Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
        print(f"Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"------------------------\n")

        annotate_originals = write_annotated_frames != "" and kwargs is not None and "original_videos" in kwargs.keys()

        unique_ids = set()
        tracked_objects = []
        
        # Iterate through frames of the video:
        fr = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # increment the count of frames
            fr +=1

            if fr <= 5:
                print(f"[FRAME DEBUG] frame {fr}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")

            # Run YOLOv8 inference
            results = model(frame * video_files[k]["amplification"], verbose=False, conf=detection_confidence, imgsz = kwargs.get("imgsz", 1280))[0].cpu()

            detections = results.boxes.xyxy.numpy()  # [x1, y1, x2, y2]
            confidences = results.boxes.conf.numpy()
            class_ids = results.boxes.cls.numpy()

            if fr <= 5:
                print(f"[YOLO DEBUG] frame {fr}: total detections = {len(detections)}")
                print(f"[YOLO DEBUG] class_ids = {class_ids}")
                print(f"[YOLO DEBUG] confidences = {confidences}")

            # Prepare detections for SORT
            # SORT expects detections in the format: [x1, y1, x2, y2, confidence]
            detections_for_sort = np.array([
                [*detection, confidence]
                for detection, confidence, cls in zip(detections, confidences, class_ids)
                if cls == 1 # only track hot bats
                and centered_in_region(detection, video_files[k]["tracking_region"], frame.shape)
            ])

            # Update SORT tracker
            objs = tracker.update(detections_for_sort)
            tracked_objects.append(objs.copy())
            
            # Count the number of unique track IDs
            for obj in tracked_objects[-1]:
                track_id = int(obj[4])
                unique_ids.add(track_id)

            if fr == 0:
                raise RuntimeError(f"No frames were read from video: {video}")

        video_basename = os.path.basename(video)

        # write annotated verson of the original video
        if annotate_originals:
            # get the name of the video file
            original_video = kwargs["original_videos"][k]
            original_basename = os.path.basename(original_video["path"])
            
            # write the annotated version of the original video.
            annotate_video(original_video["path"], os.path.join(write_annotated_frames, original_basename), tracked_objects.copy(), video_files[k]["tracking_region"])
                
            # since we're annotating the original video, we need to change the name of the debackgrounded video
            video_basename = Path(video_basename).stem + "_debackgrounded" + Path(video_basename).suffix

        # write annotated version of the debackgrounded video
        if write_annotated_frames != "":
            annotate_video(video, os.path.join(write_annotated_frames, video_basename), tracked_objects.copy(), video_files[k]["tracking_region"])



        if print_time:
            print(f"--- {round(time.time() - start_time, 2)} seconds ---\n--- {len(unique_ids)} warm bats ---\n--- {fr} frames ---\n")

        #from eric:
        # Data to store
        bg_settings = settings.get("background_subtraction", {})

        columns = ["model", "video", "bat_counts", "detection_confidence", "amplification", "tracking_region", "megadetector_threshold", "sort.max_age", "sort.min_hits", "sort.iou_threshold"]
        tracking_data = [
             model_file,
             os.path.basename(video),
             len(unique_ids),
             detection_confidence,
             video_files[k]["amplification"],
             video_files[k]["tracking_region"],
             (bg_settings | video_files[k]).get("megadetector_threshold", np.nan),
             sort_args["max_age"],
             sort_args["min_hits"],
             sort_args["iou_threshold"],
        ]
        if result_df is None:
            result_df = pd.DataFrame([tracking_data], columns=columns)
        else:
            result_df = pd.concat([pd.DataFrame([tracking_data], columns=result_df.columns), result_df], ignore_index=True)
            
    return result_df




if __name__ == "__main__":
    # Read the command-line arguments
    parser = ArgumentParser(description="Use the YAML config file to provide settings for the model.")
    parser.add_argument("-c", "--config", dest="config_path", type=str, help="Path to the config file")

    args = parser.parse_args()
    
    # Import the configuration file
    settings = get_args(args.config_path)

    # create a temporary directory for background subtraction, even if it's not needed
    with tempfile.TemporaryDirectory() as temp_dir:
        if settings.get("background_subtraction", {}).get("enabled", False):

            # remove the output_folder key from the background_subtraction settings
            if "output_folder" in settings["background_subtraction"].keys():
                settings["background_subtraction"].pop("output_folder")

            # loop through videos specified in the config file
            original_videos = check_videos(settings["tracking"]["video_files"])
            debackgrounded_videos = []

            # loop over the videos, doing background subtraction on each
            for video in original_videos:
                video_basename = os.path.basename(video["path"])
                print (f"Doing background subtraction for {video_basename}...")

                # use global background subtraction settings
                window_N = settings["background_subtraction"].get("window_size", 30)

                # original input path
                bg_input_path = video["path"]

                # convert to AVI first if needed
                if bg_input_path.lower().endswith(".mov"):
                    print(f"Converting {video_basename} to AVI for background subtraction...")
                    bg_input_path = convert_to_avi(bg_input_path, temp_dir)

                # do background subtraction
                bg_output_path = background_subtract(bg_input_path, temp_dir, window_N=window_N)

                if bg_output_path is None or not os.path.exists(bg_output_path):
                    raise FileNotFoundError(f"Background-subtracted video not created for {video['path']}")

                # expected background subtracted output path
                db_video = video.copy()
                db_video["path"] = bg_output_path
                debackgrounded_videos.append(db_video)

            # tell the model to use the modified video files, but keep track of the originals
            settings["tracking"]["video_files"] = debackgrounded_videos
            settings["tracking"]["original_videos"] = original_videos
            


        # Call the main function with the settings
        if "tracking" in settings.keys():
            result = track_bats(**settings["tracking"])

            # Save the results if the config file specifies a save_dir
            if "results_path" in settings["tracking"].keys():
                results_path = settings["tracking"]["results_path"]

                # Store results to disk
                result.to_csv(results_path, mode='a', header=not os.path.exists(results_path), index=False)
