import carla
import numpy as np
import ipdb
import h5py
import os
import open3d as o3d
def save_RGB_tensor(image_array, output_path, chunk_size=1000):
    # Assuming `data` is the original tensor of shape [N, 640, 320, 3]
    N, H, W, C = image_array.shape
    print("Original Batch Size:", N)
    # Step 1: Find the indices of frames that are not completely zero
    # We use `np.any()` along all axes except the first one (the batch axis) to determine
    # which frames have non-zero values.
    used_indices = [i for i in range(N) if np.any(image_array[i])]
    # Step 2: Trim the tensor using the identified indices
    # This will give you a tensor of shape [T, 640, 320, 3], where T < N
    trimmed_data = image_array[used_indices]
    print("Trimmed Batch Size:", trimmed_data.shape[0])
    # Save data in chunks
    num_chunks = trimmed_data.shape[0] // chunk_size
    print(f"Saving {num_chunks} Chunks")
    remainder = trimmed_data.shape[0] % chunk_size
    chunks = {}
    # Save the evenly divisible chunks
    for i in range(num_chunks):
        chunk = trimmed_data[i * chunk_size : (i + 1) * chunk_size]
        chunks[f'data_chunk_{i}'] = chunk  # Add each chunk to the dictionary
    # Handle the remaining data as the last chunk
    if remainder > 0:
        last_chunk = trimmed_data[num_chunks * chunk_size :]
        chunks[f'data_chunk_{num_chunks}'] = last_chunk
        # Save all chunks to a single .npz file
    np.savez(output_path, **chunks)
    return 0
        
def process_semSeg(image):
    """
    Process the CARLA segmentation image to map classes to a single road category
    and produce a binary mask NumPy array.
    
    Args:
        image (carla.Image): CARLA Image object from the semantic segmentation sensor.

    Returns:
        np.ndarray: Binary mask array where road and roadlines are 1, and all else is 0.
    """

    # Define class IDs based on CARLA's segmentation settings
    ROAD_CLASS = 1       
    ROADLINE_CLASS = 24   
    OTHER_CLASS = 0      # Mask other classes as black

    # Convert raw image data to a NumPy array (assumes 4 channels: BGRA)
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    # Extract the red channel which contains the class IDs
    class_map = array[:, :, 2]  # Red channel

    # Create binary mask: Merge road and roadline classes, assign all else to 'other'
    binary_mask = np.where(
        (class_map == ROAD_CLASS) | (class_map == ROADLINE_CLASS),
        1,
        0
    ).astype(np.uint8)
    return binary_mask


    
def process_image(image):
    # Convert and process image as before
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]       # Remove alpha channel
    array = array[:, :, ::-1]     # Convert BGRA to RGB
    return array

def process_lidar(data, o3d_vis=None, o3d_pc=None):
    # Convert and process LiDAR data as before
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (-1, 4))
    points = points[~np.isnan(points).any(axis=1)] #delete points with NaN
    if o3d_vis is not None and o3d_pc is not None:
        # Extract XYZ coordinates from the frame
        xyz_points = points[:, :3]
        xyz_points[:,0] = -xyz_points[:,0]
        # Create a PointCloud object and assign the points
        o3d_pc.points = o3d.utility.Vector3dVector(xyz_points)
        # Apply a bounding box to focus the view on the points
        o3d_vis.clear_geometries()
        o3d_vis.add_geometry(o3d_pc)
        o3d_vis.update_geometry(o3d_pc)
        o3d_vis.poll_events()
        o3d_vis.update_renderer()
    return points
    
def process_imu(data):
    # Extract IMU data
    accel_data = np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z], dtype=np.float32)  # carla.Vector3D (m/s^2)
    gyro_data = np.array([data.gyroscope.x, data.gyroscope.y, data.gyroscope.z], dtype=np.float32)     # carla.Vector3D (rad/s)
    return [accel_data, gyro_data]


def save_lidar_sequence(lidar_data, output_path, chunk_size=100, timestamp = None, metadata=None):
    """
    Save a sequence of LIDAR frames with varying point counts in chunks.
    
    Args:
        lidar_data: List of numpy arrays, where each array has shape [N, 4] with N varying.
        output_path: Directory path to save the .npz files for chunks.
        chunk_size: Number of frames per chunk.
        timestamp: Optional timestamp to include in the filenames.
        metadata: Optional dictionary of metadata to save.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Split lidar_data into chunks
    num_frames = len(lidar_data)
    num_chunks = (num_frames + chunk_size - 1) // chunk_size  # Calculate total chunks

    for chunk_idx in range(num_chunks):
        # Get the current chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_frames)
        chunk_data = lidar_data[start_idx:end_idx]

        # Prepare data for the current chunk
        chunk_dict = {
            'lidar_frames': np.array(chunk_data, dtype=object),
            'num_frames': len(chunk_data),
            'frame_sizes': np.array([frame.shape[0] for frame in chunk_data]),
        }

        # Add metadata to each chunk, if provided
        if metadata is not None:
            chunk_dict.update(metadata)

        # Create a unique file name for the chunk
        chunk_filename = f"lidar_chunk_{chunk_idx:03d}"
        if timestamp:
            chunk_filename += f"_{timestamp}"
        chunk_filename += ".npz"

        # Save the chunk
        np.savez_compressed(os.path.join(output_path, chunk_filename), **chunk_dict)

    print(f"Saved {num_chunks} chunks to {output_path}")