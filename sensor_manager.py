import carla
import numpy as np
import pygame
import argparse
import random
import time

def process_semSeg(image):
    """
    Process the CARLA segmentation image to map classes to three categories
    and produce a color-coded NumPy array for visualization.
    
    Args:
        image (carla.Image): CARLA Image object from the semantic segmentation sensor.

    Returns:
        np.ndarray: RGB image array where each class is color-coded.
    """

    # Define class IDs based on CARLA's segmentation settings
    ROAD_CLASS = 1       
    ROADLINE_CLASS = 24   
    OTHER_CLASS = 22    

    # Define colors for visualization (in RGB format)
    CLASS_COLORS = {
        ROAD_CLASS: [128, 64, 128],      # Example color for roads (purple)
        ROADLINE_CLASS: [157, 234, 50],  # Example color for roadlines (greenish)
        OTHER_CLASS: [0, 0, 0]           # Example color for other (black)
    }
    # Convert raw image data to a NumPy array (assumes 4 channels: BGRA)
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    # Extract the red channel which contains the class IDs
    class_map = array[:, :, 2]  # Red channel
    # Map classes: Keep only road, roadline, and assign all else to 'other'
    processed_map = np.where(
        (class_map == ROAD_CLASS),
        class_map,
        OTHER_CLASS
    )
    
    # Initialize the RGB output image
    color_image = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    # Apply colors based on the class ID
    for class_id, color in CLASS_COLORS.items():
        color_image[processed_map == class_id] = color

    return color_image
    
def process_image(image):
    # Convert and process image as before
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]       # Remove alpha channel
    array = array[:, :, ::-1]     # Convert BGRA to RGB
    return array

def process_lidar(data):
    # Convert and process LiDAR data as before
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (-1, 4))
    return points
    
def process_imu(data):
    # Extract IMU data
    accel_data = np.array([data.accelerometer.x, data.accelerometer.y, data.accelerometer.z])  # carla.Vector3D (m/s^2)
    gyro_data = np.array([data.gyroscope.x, data.gyroscope.y, data.gyroscope.z])     # carla.Vector3D (rad/s)
    return [accel_data, gyro_data]

