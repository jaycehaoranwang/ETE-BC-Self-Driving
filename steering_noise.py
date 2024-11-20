import numpy as np
import matplotlib.pyplot as plt
import random

def generate_temporally_correlated_noise(duration, dt, max_noise_magnitude, max_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    num_steps = int(duration / dt)
    noise = np.zeros(num_steps)
    
    # Generate random rate of change for noise
    rate = np.random.uniform(0.005, max_rate)
    direction = np.random.choice([-1, 1])
    target_value = np.random.uniform(-max_noise_magnitude, max_noise_magnitude)
    
    current_value = 0
    
    for i in range(num_steps):
        # Introduce more frequent direction changes
        if random.uniform(0, 1) < 0.05:  # 10% chance to change direction at each step
            direction *= -1

        if direction > 0:
            current_value += rate * dt
            if current_value >= target_value:
                direction = -1
        else:
            current_value -= rate * dt
            if current_value <= -target_value:
                direction = 1
                # Change target value and rate randomly after each complete cycle
                target_value = np.random.uniform(-max_noise_magnitude, max_noise_magnitude)
                rate = np.random.uniform(0.005, max_rate)
        
        # Clip the noise value to the desired maximum magnitude
        current_value = np.clip(current_value, -max_noise_magnitude, max_noise_magnitude)
        noise[i] = current_value
    
    # Ensure the mean of the noise over time is approximately zero
    noise -= np.mean(noise)
    
    return noise

if __name__ == "__main__":
    # Example Usage
    duration = 600.0  # seconds
    dt = 0.05  # seconds per step
    max_noise_magnitude = 0.1  # Maximum steering noise magnitude
    max_rate = 0.05  # Maximum rate of change per second

    steering_noise = generate_temporally_correlated_noise(duration, dt, max_noise_magnitude, max_rate, seed=322)

    # Plot to visualize the generated noise
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(0, duration, dt), steering_noise, label="Steering Noise")
    plt.xlabel("Time (s)")
    plt.ylabel("Steering Noise")
    plt.title("Temporally Correlated Steering Noise")
    plt.legend()
    plt.grid()
    plt.show()
