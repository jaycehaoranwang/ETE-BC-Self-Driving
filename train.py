import os
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import open3d as o3d
import time
import cv2
from models import PointPillars

def visualize_point_cloud(lidar_frame):
    """
    Visualizes a single point cloud using Open3D.

    Args:
        lidar_frame: Numpy array of shape [N, 4] representing the point cloud,
                     where each row is (x, y, z, intensity).
    """
    # Ensure that the input frame has points
    if lidar_frame.shape[0] == 0:
        print("The input point cloud is empty. Nothing to visualize.")
        return

    # Extract XYZ coordinates from the frame
    xyz_points = lidar_frame[:, :3]

    # Create a PointCloud object and assign the points
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz_points)

    # Apply a bounding box to focus the view on the points
    o3d.visualization.draw_geometries([point_cloud], 
                                      zoom=0.5,
                                      front=[0, 0, -1],
                                      lookat=point_cloud.get_center(),
                                      up=[0, -1, 0])

def load_lidar_sequence(input_path):
    """
    Load a sequence of LIDAR frames saved in .npz chunks.

    Args:
        input_path: Directory containing the .npz files.

    Returns:
        lidar_data: List of numpy arrays, where each array has shape [N, 4].
    """
    lidar_data = []  # List to hold all frames

    # Get a sorted list of all .npz files in the directory
    npz_files = sorted([f for f in os.listdir(input_path) if f.endswith('.npz')])
    for npz_file in npz_files:
        # Load the .npz file
        file_path = os.path.join(input_path, npz_file)
        with np.load(file_path, allow_pickle=True) as data:
            # Extract LIDAR frames and add to the sequence
            lidar_frames = data['lidar_frames']
            lidar_data.extend(lidar_frames)

    print(f"Loaded {len(lidar_data)} frames from {len(npz_files)} chunks.")
    return lidar_data

def load_control_data(input_path):
    """
    Loads a saved control data from an .npz file.

    Args:
        input_path: Path to the .npz file containing the saved control data.

    Returns:
        control_data: Numpy array of shape [T, 3], where T is the total number of frames.
    """
    # Load the .npz file
    with np.load(input_path) as data:
        # Extract the control data
        steering_data = data['steering_input']
        driving_state_data = data['driving_state']
        noisy_steering_data = data['noisy_steering_input']
    return steering_data, noisy_steering_data, driving_state_data

def load_RGB_tensor(input_path):
    """
    Loads a saved image tensor from an .npz file.

    Args:
        input_path: Path to the .npz file containing the saved image chunks.

    Returns:
        image_array: Numpy array of shape [T, H, W, C], where T is the total number of frames.
    """
    # Load the .npz file
    with np.load(input_path) as data:
        # Extract all the chunks and concatenate them to reconstruct the original tensor
        chunks = [data[key] for key in data.keys()]
        image_array = np.concatenate(chunks, axis=0)
    print("Loaded Image Array Shape:", image_array.shape)
    return image_array

def display_RGB_images(image_array):
    """
    Displays RGB images in sequence with a 0.1 second pause between each frame.

    Args:
        image_array: Numpy array of shape [T, H, W, C], representing RGB images.
    """
    # Loop through each frame in the tensor and display
    for i in range(image_array.shape[0]):
        # Get the current frame
        image = image_array[i]

        # Convert the image from RGB to BGR for OpenCV display
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image using OpenCV
        cv2.imshow("RGB Image Sequence", image_bgr)

        # Pause for 0.1 seconds
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # Alternatively, use time.sleep() if cv2.waitKey() doesn't work well with your setup
        # time.sleep(0.1)

    # Close the window when finished
    cv2.destroyAllWindows()

def display_lidar_sequence(lidar_data, o3d_vis, o3d_pc, loop=False):
    """
    Displays a sequence of LIDAR frames using Open3D.

    Args:
        lidar_data: List of numpy arrays, where each array has shape [N, 4].
    """
    # Set up Open3D Visualizer
    o3d_vis.create_window()
    o3d_pc = o3d.geometry.PointCloud()
    o3d_vis.add_geometry(o3d_pc)

    if loop:
        while True:
            # Loop through each frame in the sequence
            for frame in lidar_data:
                # Remove NaN values from the frame
                point_cloud = frame
                # Create a PointCloud object and assign the points
                o3d_pc.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                # Apply a bounding box to focus the view on the points
                o3d_vis.clear_geometries()
                o3d_vis.add_geometry(o3d_pc)
                o3d_vis.update_geometry(o3d_pc)
                o3d_vis.poll_events()
                o3d_vis.update_renderer()
    else:
        # Loop through each frame in the sequence
        for frame in lidar_data:
            # Remove NaN values from the frame
            point_cloud = frame
            # Create a PointCloud object and assign the points
            o3d_pc.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            # Apply a bounding box to focus the view on the points
            o3d_vis.clear_geometries()
            o3d_vis.add_geometry(o3d_pc)
            o3d_vis.update_geometry(o3d_pc)
            o3d_vis.poll_events()
            o3d_vis.update_renderer()
    # Close the Open3D window when finished
    o3d_vis.destroy_window()

# Example usage
if __name__ == "__main__":
    input_path = r"expert_data\Town01\20241129_041828\lidar_sequence_20241129_041828_steerNoiseCap_15"
    lidar_sequence = load_lidar_sequence(input_path)
    #steering, noisy_steering, driving_state = load_control_data(r"expert_data\Town01\20241129_041828\control_data_20241129_041828_steerNoiseCap_15.npz")
    #rgb_images = load_RGB_tensor(r"expert_data\Town01\20241129_030720\rgb_data_20241129_030720_steerNoiseCap_2.npz")
    #display_RGB_images(rgb_images)
    
    #process point cloud frame by frame
    # Set up Open3D Visualizer
    ipdb.set_trace()
    o3d_vis = o3d.visualization.Visualizer()
    o3d_vis.create_window()
    o3d_pc = o3d.geometry.PointCloud()
    o3d_vis.add_geometry(o3d_pc)
    display_lidar_sequence(lidar_sequence, o3d_vis, o3d_pc)

    # Example: Print some details about the loaded data
    

