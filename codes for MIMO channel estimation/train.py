import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Function to generate THz MIMO data (can be imported from your existing code)
def generate_thz_mimo_data(num_samples=10000, num_antennas=256, num_users=10, noise_power=1e-2, save_file=False):
    los_component = np.random.randn(num_samples, num_antennas, num_users)
    nlos_component = np.random.randn(num_samples, num_antennas, num_users) * 0.1
    channel_matrix = los_component + nlos_component
    noise = np.sqrt(noise_power) * np.random.randn(num_samples, num_antennas, num_users)
    noisy_channel_matrix = channel_matrix + noise

    X_train, X_temp, y_train, y_temp = train_test_split(noisy_channel_matrix, channel_matrix, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save dataset as .npz file if save_file=True
    if save_file:
        np.savez_compressed('thz_mimo_dataset.npz', X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Define linear and non-linear models
def linear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512, activation='linear'))
    model.add(layers.Dense(np.prod(input_shape), activation='linear'))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def nonlinear_model(input_shape):
    model = models.Sequential()
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))  # Reshaping for Conv2D
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))  # Reduce number of filters
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))  # Reduce units here
    model.add(layers.Dense(np.prod(input_shape), activation='linear'))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train and save the models
def train_model(model, X_train, y_train, X_val, y_val, model_name='model'):
    # Create a checkpoint to save the best weights
    checkpoint = ModelCheckpoint(f'{model_name}_best_weights.keras', save_best_only=True, monitor='val_loss', mode='min')
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint])
    
    # Save the final model
    model.save(f'{model_name}_final_model.keras')
    
    return history

if __name__ == "__main__":
    print("Generating dataset...")
    # Generate and save the dataset (change save_file=True to store the data)
    X_train, X_val, X_test, y_train, y_val, y_test = generate_thz_mimo_data(save_file=True)

    print("Training linear model...")
    # Create and train linear model
    linear_estimation_model = linear_model(X_train.shape[1:])
    train_model(linear_estimation_model, X_train, y_train, X_val, y_val, model_name='linear_model')

    print("Training non-linear model...")
    # Create and train non-linear model
    nonlinear_estimation_model = nonlinear_model(X_train.shape[1:])
    train_model(nonlinear_estimation_model, X_train, y_train, X_val, y_val, model_name='nonlinear_model')