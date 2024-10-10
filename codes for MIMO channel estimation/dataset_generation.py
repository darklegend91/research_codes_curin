# dataset_generation.py
import numpy as np
from sklearn.model_selection import train_test_split

def generate_thz_mimo_data(num_samples=10000, num_antennas=256, num_users=10, noise_power=1e-2):
    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component
    noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
    noisy_channel_matrix = channel_matrix + noise

    X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test