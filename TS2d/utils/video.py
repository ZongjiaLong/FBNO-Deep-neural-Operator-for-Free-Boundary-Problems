import os
import glob
import cv2
import numpy as np


def create_video_from_pngs(png_folder, output_video, fps=5, target_width=None):
    """
    Create a video from PNG images in a folder.

    Args:
        png_folder: Path to folder containing PNG images
        output_video: Path for output video file
        fps: Frames per second (default: 5)
        target_width: Target width for video (None for original size)
    """
    # Find all PNG files in the folder and sort them
    png_files = sorted(glob.glob(os.path.join(png_folder, "*.png")))

    if not png_files:
        print("No PNG files found")
        return

    print(f"Found {len(png_files)} PNG files")

    # Read the first frame to get dimensions
    first_frame = cv2.imread(png_files[0])
    if first_frame is None:
        print("Cannot read the first frame")
        return

    # Get original dimensions
    original_height, original_width = first_frame.shape[:2]

    # Calculate target dimensions
    if target_width is None:
        # Keep original dimensions
        target_width = original_width
        target_height = original_height
    else:
        # Calculate proportional height based on target width
        target_height = int(target_width * original_height / original_width)
        # Ensure height is even (some codecs require even dimensions)
        if target_height % 2 != 0:
            target_height += 1

    # Video dimensions
    size = (target_width, target_height)
    print(f"Target video dimensions: {target_width} x {target_height}")
    print(f"Frame rate: {fps} fps")
    print(f"Estimated video duration: {len(png_files) / fps:.2f} seconds")

    # Initialize video writer (MP4 codec)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, size)

    # Process each PNG file
    for i, png_file in enumerate(png_files):
        # Read frame
        frame = cv2.imread(png_file)

        if frame is None:
            print(f"Warning: Cannot read frame {i}: {png_file}")
            continue

        # Resize if dimensions don't match
        if frame.shape[1] != target_width or frame.shape[0] != target_height:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        # Write frame to video
        out.write(frame)

    # Release video writer
    out.release()
    print()  # Empty line for better readability

    # Verify and display video information
    if os.path.exists(output_video):
        file_size = os.path.getsize(output_video) / (1024 * 1024)  # Convert to MB
        print(f"✓ Video successfully created: {output_video}")
        print(f"✓ File size: {file_size:.2f} MB")
        print(f"✓ Resolution: {target_width} x {target_height}")
        print(f"✓ Frame rate: {fps} fps")
        print(f"✓ Total frames: {len(png_files)}")
        print(f"✓ Total duration: {len(png_files) / fps:.2f} seconds")
    else:
        print(f"✗ Error: Failed to create video file")


# Usage example
if __name__ == "__main__":
    # Configuration parameters
    png_folder = "D:\\desktop\\output861\\visualization_cx_0.85_cy_0.60"
    output_video = "D:\\desktop\\output861\\visualization_cx_0.85_cy_0.60\\output_video.mp4"

    # Check if folder exists
    if not os.path.exists(png_folder):
        print(f"Error: Folder {png_folder} does not exist")
    else:
        # Create video
        create_video_from_pngs(
            png_folder=png_folder,
            output_video=output_video,
            fps=5,  # Frame rate
            target_width=2400  # Video width
        )