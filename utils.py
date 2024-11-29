import carla
import pygame
import numpy as np
import random
import os
import queue
from time import time
import cv2

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, ego_vehicle, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.ego_vehicle = ego_vehicle
    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        velocity = self.ego_vehicle.get_velocity()
        return data, velocity

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
            

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def print_vehicle_info(vehicle): 
    # Get vehicle information
    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s to km/h
    info_text = f"Vel_x:{velocity.x:.2f} m/s, Vel_y: {velocity.y:.2f} m/s, Vel_z:{velocity.z:.2f} m/s, Speed: {speed:.2f} km/h"
    print(info_text)

def steer_const_speed(vehicle, curr_speed, set_speed, steer):
    control = vehicle.get_control()
    control.throttle = 0.5 if curr_speed < set_speed else 0.0
    control.brake = 0.5 if curr_speed > set_speed else 0.0
    control.steer = steer
    vehicle.apply_control(control)

def keyboard_control(vehicle):
    control = carla.VehicleControl()
    keys = pygame.key.get_pressed()

    # Set throttle, brake, steer based on key inputs
    if keys[pygame.K_w]:  # Forward
        control.throttle = 1.0
    elif keys[pygame.K_s]:  # Reverse
        control.reverse = True
        control.throttle = 1.0

    if keys[pygame.K_a]:  # Left
        control.steer = -0.5
    elif keys[pygame.K_d]:  # Right
        control.steer = 0.5

    if keys[pygame.K_SPACE]:  # Brake
        control.brake = 1.0

    # Apply the control to the vehicle
    vehicle.apply_control(control)

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def visualize_lidar_top_down(points, img_size=600, scale=10):
    """
    Visualizes a top-down 2D view of LiDAR points using OpenCV.
    
    Parameters:
    - points: N x 4 numpy array where each row is [x, y, z, intensity].
    - img_size: Size of the square image to visualize.
    - scale: Scale factor to adjust the spread of points in the visualization.
    
    Each time this function is called, it displays the points on a new frame.
    """
    # Initialize a blank image
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Center of the image for the origin (0, 0)
    center_x, center_y = img_size // 2, img_size // 2
    
    # Normalize the intensity to a range from 0 to 255
    intensities = np.clip(points[:, 3] * 255, 0, 255).astype(np.uint8)
    
    for i in range(points.shape[0]):
        # Scale the x and y coordinates and shift them to the image center
        x = int(center_x + points[i, 0] * scale)
        y = int(center_y - points[i, 1] * scale)  # Flip y-axis for top-down view
        
        # Check if the point falls within the image bounds
        if 0 <= x < img_size and 0 <= y < img_size:
            # Draw the point using intensity as the color (greyscale)
            img[y, x] = (intensities[i], intensities[i], intensities[i])
    
    # Display the image
    cv2.imshow("Top-Down LiDAR View", img)
    cv2.waitKey(1)  # 1 ms delay for continuous updating

def start_carla_server():
    # Modify the path to the CARLA executable if necessary
    carla_path = 'X:\\CarlaUE4\\CarlaUE4'  # Change this path
    if os.name == 'posix':  # Unix-based systems
        carla_path += '.sh'
    elif os.name == 'nt':   # Windows
        carla_path += '.exe'
    
    os.system(f'{carla_path} -quality-level=Epic &')
    time.sleep(5)  # Give the server time to initialize

def initialize_carla_client(host='127.0.0.1', port=2000, timeout=10):
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    print(f"Connected to CARLA server at {host}:{port}")
    return client

def spawn_vehicle(world, blueprint_filter="vehicle.tesla.model3", spawn_point=None, ego=True):
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find(blueprint_filter)
    vehicle_bp.set_attribute('color', '255,255,255')
    if ego:
        vehicle_bp.set_attribute('role_name', 'hero')
    spawn_point = spawn_point or random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Spawned vehicle: {vehicle.type_id} at {spawn_point.location}")
    return vehicle

def attach_camera(world, vehicle, camera_params):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    for attr_name, attr_value in camera_params.items():
        if camera_bp.has_attribute(attr_name):
            camera_bp.set_attribute(attr_name, str(attr_value))
        else:
            print(f"RGB Camera: Attribute {attr_name} not found in blueprint.")

    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)), attach_to=vehicle)
    return camera

def attach_lidar(world, vehicle, lidar_params):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')

    for attr_name, attr_value in lidar_params.items():
        if lidar_bp.has_attribute(attr_name):
            lidar_bp.set_attribute(attr_name, str(attr_value))
        else:
            print(f"LIDAR: Attribute {attr_name} not found in blueprint.")


    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0, z=3.4), carla.Rotation(yaw=+00)), attach_to=vehicle)
    return lidar

def attach_IMU(world, ego_vehicle, imu_params):
    imu_bp = world.get_blueprint_library().find('sensor.other.imu')

    for attr_name, attr_value in imu_params.items():
        if imu_bp.has_attribute(attr_name):
            imu_bp.set_attribute(attr_name, str(attr_value))
        else:
            print(f"IMU: Attribute {attr_name} not found in blueprint.")

    # Sensor is placed at the origin of vehicle coordinate frame (adjust if needed)
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego_vehicle)

def visualize_depth(depth_map, window_name="Depth Map"):
    """
    Visualize a depth map using OpenCV with normalization and a colormap.

    Parameters:
    - depth_map: np.ndarray
        HxW numpy array representing the depth map.
    - window_name: str
        The name of the visualization window.

    Returns:
    - None
    """
    # Normalize depth map to 0-255 for visualization
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to 8-bit for proper display
    depth_8bit = depth_normalized.astype(np.uint8)
    # Apply a colormap (e.g., COLORMAP_JET or COLORMAP_MAGMA)
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    # Display the depth map
    cv2.imshow(window_name, depth_colormap)
    cv2.waitKey(1)  # Display the frame for a short period (1ms) to allow updates

    # Function to update the image
def visualize_rgb(new_image_data):
    disp_img = new_image_data * 255
    cv2.imshow('Segmentation Map', disp_img)
    cv2.waitKey(1)  # Display the image and wait for 1 millisecond (allows real-time updates)