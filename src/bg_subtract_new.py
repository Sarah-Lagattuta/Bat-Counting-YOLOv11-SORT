import cv2
import numpy as np
import os
from collections import deque


def background_subtract(video_path, output_dir, window_N=10):
    # Open the input video
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare output filename
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_bg_subtract.avi")

    # Video writer object
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (frame_width, frame_height),
        isColor=False,
    )

    frame_buffer = deque(maxlen=window_N)

    gamma = 0.5
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")

    # Load initial window_N frames into buffer
    for _ in range(window_N):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Not enough frames for initial window in {video_path}")
            cap.release()
            out.release()
            return None
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(gray_frame)

    # Reset the video to start from the first frame again
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute background (median of frames in buffer)
        background = np.median(np.stack(frame_buffer), axis=0).astype(np.uint8)

        # Background subtraction and gamma adjustment
        fg_mask = cv2.absdiff(gray_frame, background)
        fg_mask = cv2.LUT(fg_mask, table)

        # Write frame to output
        out.write(fg_mask)

        # Update the buffer
        frame_buffer.append(gray_frame)

    cap.release()
    out.release()

    print(f"Background subtraction complete. Saved to {output_path}")
    return output_path
