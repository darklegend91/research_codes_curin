import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('thz_mimo_dataset.npz')
X_test = data['X_test']
y_test = data['y_test']

# Load the trained models
linear_model = load_model('linear_model_final_model.keras')
nonlinear_model = load_model('nonlinear_model_final_model.keras')

# Define a function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 10 * np.log10(mse / norm_factor)

# Evaluate across different iterations
iterations = [1, 5, 10, 15, 20]
nmse_linear = []
nmse_nonlinear = []

for iteration in iterations:
    # Add noise according to the iteration count
    noise = np.random.normal(0, 0.1, X_test.shape) * iteration
    X_test_noisy = X_test + noise
    
    # Make predictions
    y_pred_linear = linear_model.predict(X_test_noisy)
    y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)
    
    # Compute NMSE
    nmse_linear.append(compute_nmse(y_test, y_pred_linear))
    nmse_nonlinear.append(compute_nmse(y_test, y_pred_nonlinear))

# Plot the results
plt.plot(iterations, nmse_linear, 'o-', label='Linear Model NMSE')
plt.plot(iterations, nmse_nonlinear, 's-', label='Nonlinear Model NMSE')
plt.xlabel('Iterations')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid(True)
plt.title('NMSE vs Number of Iterations')

# Save the plot
plt.savefig('nmse_vs_iterations.png')

# Show the plot
plt.show()

print("Chart saved as 'nmse_vs_iterations.png'")
