# train_and_test.py
import numpy as np
from dataset_generation import generate_thz_mimo_data
from model_definition import linear_model, nonlinear_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, X_train, y_train, X_val, y_val, model_name):
    # Use '.keras' extension for saving weights as per the new Keras format
    checkpoint = ModelCheckpoint(f'{model_name}_best_weights.keras', save_best_only=True, monitor='val_loss', mode='min')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[checkpoint]
    )
    return history

def test_model(model, X_test, y_test, model_name):
    # Load the best weights saved during training
    model.load_weights(f'{model_name}_best_weights.keras')
    
    # Evaluate model performance on the test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Predict the channel matrix using the trained model
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error (MSE) between predicted and true values
    mse = np.mean(np.square(y_pred - y_test))
    print(f"{model_name} Test MSE: {mse}")
    
    return y_pred

if __name__ == "__main__":
    # Generate the dataset for THz Ultra-Massive MIMO channel estimation
    X_train, X_val, X_test, y_train, y_val, y_test = generate_thz_mimo_data()

    # Initialize the linear and non-linear models
    linear_estimation_model = linear_model(X_train.shape[1:])
    nonlinear_estimation_model = nonlinear_model((X_train.shape[1], X_train.shape[2], 1))

    # Train the linear model and save the best weights
    history_linear = train_model(linear_estimation_model, X_train, y_train, X_val, y_val, model_name='linear_model')

    # Train the non-linear model and save the best weights
    history_nonlinear = train_model(nonlinear_estimation_model, X_train, y_train, X_val, y_val, model_name='nonlinear_model')

    # Test the linear model and print the results
    y_pred_linear = test_model(linear_estimation_model, X_test, y_test, model_name='linear_model')

    # Test the non-linear model and print the results
    y_pred_nonlinear = test_model(nonlinear_estimation_model, X_test, y_test, model_name='nonlinear_model')

    # Compare the results for both models
    print(f"Linear model MAE on test set: {np.mean(np.abs(y_pred_linear - y_test))}")
    print(f"Non-linear model MAE on test set: {np.mean(np.abs(y_pred_nonlinear - y_test))}")