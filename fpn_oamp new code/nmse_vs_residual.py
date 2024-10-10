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

# Define a function to compute residual norm (difference between prediction and true)
def compute_residual_norm(y_true, y_pred):
    residual = y_true - y_pred
    norm = np.linalg.norm(residual.flatten())
    return norm

# Evaluate residual norm across different SNR levels
snr_levels = [0, 5, 10, 15, 20]
residual_norm_linear = []
residual_norm_nonlinear = []

for snr in snr_levels:
    noise = np.random.normal(0, 10**(-snr / 20), X_test.shape)
    X_test_noisy = X_test + noise
    
    # Make predictions
    y_pred_linear = linear_model.predict(X_test_noisy)
    y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)
    
    # Compute residual norm
    residual_norm_linear.append(compute_residual_norm(y_test, y_pred_linear))
    residual_norm_nonlinear.append(compute_residual_norm(y_test, y_pred_nonlinear))

# Plot the results
plt.plot(snr_levels, residual_norm_linear, 'o-', label='Linear Model Residual Norm')
plt.plot(snr_levels, residual_norm_nonlinear, 's-', label='Nonlinear Model Residual Norm')
plt.xlabel('SNR (dB)')
plt.ylabel('Residual Norm')
plt.legend()
plt.grid(True)
plt.title('Residual Norm vs SNR')

# Save the chart
plt.savefig('residual_norm_vs_snr.png')

# Show the plot
plt.show()

print("Chart saved as 'residual_norm_vs_snr.png'")
