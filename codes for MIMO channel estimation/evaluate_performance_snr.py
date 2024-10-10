import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# Load the dataset
data = np.load('thz_mimo_dataset.npz')
X_test = data['X_test']
y_test = data['y_test']

# Load the trained linear and non-linear models
linear_model = load_model('linear_model_final_model.keras')
nonlinear_model = load_model('nonlinear_model_final_model.keras')

# Define a function to compute NMSE (Normalized Mean Squared Error)
def compute_nmse(y_true, y_pred):
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    norm_factor = np.linalg.norm(y_true.flatten()) ** 2
    return 10 * np.log10(mse / norm_factor)

# Evaluate across different SNR levels (assuming you add noise)
snr_levels = [0, 5, 10, 15, 20]
nmse_linear = []
nmse_nonlinear = []

for snr in snr_levels:
    # Add noise according to SNR level
    noise = np.random.normal(0, 10**(-snr / 20), X_test.shape)
    X_test_noisy = X_test + noise
    
    # Make predictions
    y_pred_linear = linear_model.predict(X_test_noisy)
    y_pred_nonlinear = nonlinear_model.predict(X_test_noisy)
    
    # Compute NMSE
    nmse_linear.append(compute_nmse(y_test, y_pred_linear))
    nmse_nonlinear.append(compute_nmse(y_test, y_pred_nonlinear))

# Plot the results
plt.plot(snr_levels, nmse_linear, 'o-', label='Linear Model NMSE')
plt.plot(snr_levels, nmse_nonlinear, 's-', label='Nonlinear Model NMSE')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid(True)
plt.title('NMSE vs SNR')

# Save the plot
plt.savefig('nmse_vs_snr.png')

# Show the plot
plt.show()

print("Chart saved as 'nmse_vs_snr.png'")