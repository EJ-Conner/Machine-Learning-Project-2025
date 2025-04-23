
import os
import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt



tf_imported = False
Sequential, SimpleRNN, Dense, Dropout, ModelCheckpoint = None, None, None, None, None
try:
    # Try importing base tensorflow first to catch the DLL error early if RNN is selected
    import tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint
    tf_imported = True
    print("TensorFlow imported successfully.")
except ImportError as e:
    print(f"Warning: TensorFlow could not be imported. RNN model will not be available.")
    print(f"Import Error: {e}")
    # Keep placeholders as None
except Exception as e:
    print(f"An unexpected error occurred during TensorFlow import: {e}")
    traceback.print_exc()


def convert_volume(vol_str):
    """Converts volume strings (e.g., '121.90K', '691.49M') to float."""
    if isinstance(vol_str, (int, float)): # Return if already numeric
        return float(vol_str)
    vol_str = str(vol_str).strip().upper() # Ensure string, remove whitespace, handle case
    
    if 'B' in vol_str:
        return float(vol_str.replace('B', '')) * 1_000_000_000
    elif 'M' in vol_str:
        return float(vol_str.replace('M', '')) * 1_000_000
    elif 'K' in vol_str:
        return float(vol_str.replace('K', '')) * 1_000
    else:
        try:
            # Handle cases with no suffix (just a number)
            return float(vol_str)
        except ValueError:
            # Handle unexpected formats or potential NaNs gracefully
            print(f"Warning: Could not convert volume value: {vol_str}. Returning NaN.")
            return np.nan

#Base Preprocessing Class
class Preprocessing:

    def __init__(self, dataset_path):
        #Load the dataset
        try:
            self.df = pd.read_csv(dataset_path)
            print(f"Dataset loaded successfully from {dataset_path}")
            # Convert 'Date' column to datetime objects right after loading
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            # Sort by date to ensure correct lagging and identification of the last row
            self.df = self.df.sort_values(by='Date').reset_index(drop=True)
        except FileNotFoundError:
            print(f"Error: Dataset file not found at {dataset_path}")
            raise
        except Exception as e:
            print(f"Error loading or parsing dataset: {e}")
            raise

        self.scaler = None # To store the feature scaler
        self.last_original_features_for_prediction = None # To store the row for future prediction
        self.feature_names = None # Store feature names after preprocessing

    def preprocess_data(self):

        # --- Data Cleaning and Type Conversion ---
        print("Starting data cleaning and type conversion...")
        # Handle 'Change %' - remove '%', convert to float, divide by 100
        if 'Change %' in self.df.columns:
            try:
                self.df['Change %'] = self.df['Change %'].astype(str).str.replace('%', '', regex=False)
                self.df['Change %'] = pd.to_numeric(self.df['Change %'], errors='coerce') / 100.0
                
            except Exception as e:
                print(f"Error processing 'Change %': {e}. Skipping column.")
                self.df.drop('Change %', axis=1, inplace=True, errors='ignore') # Drop if problematic
        else:
            print("Column 'Change %' not found.")


        # Handle 'Vol.' using the convert_volume function
        if 'Vol.' in self.df.columns:
            original_vol_type = self.df['Vol.'].dtype
            print(f"Processing 'Vol.' column (original type: {original_vol_type})...")
            self.df['Vol.'] = self.df['Vol.'].apply(convert_volume)
            # Check conversion results
            if self.df['Vol.'].isnull().any():
                 print(f"Warning: Some 'Vol.' values resulted in NaN after conversion.")
            print("'Vol.' processed.")
        else:
             print("Column 'Vol.' not found.")


        # Handle Price, Open, High, Low - remove commas, convert to float
        price_cols = ['Price', 'Open', 'High', 'Low']
        for col in price_cols:
            if col in self.df.columns:
                try:
                    # Ensure the column is treated as string before replacing commas
                    self.df[col] = self.df[col].astype(str).str.replace(',', '', regex=False)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except Exception as e:
                    print(f"Error processing column '{col}': {e}. Skipping column.")
                    self.df.drop(col, axis=1, inplace=True, errors='ignore') # Drop if problematic
            else:
                print(f"Column '{col}' not found.")

        # --- Feature Engineering (Lags) ---
        print("Creating lagged features...")
        # Define columns to lag (ensure they exist after cleaning)
        cols_to_lag = [col for col in ['Price', 'Change %', 'Vol.', 'High', 'Low', 'Open'] if col in self.df.columns]
        if not cols_to_lag:
             raise ValueError("No suitable columns found to create lagged features after cleaning.")

        n_lags = 5
        for i in range(1, n_lags + 1):
            for col in cols_to_lag:
                 self.df[f'{col}_lag_{i}'] = self.df[col].shift(i)
        print(f"Created lags up to {n_lags} days for columns: {cols_to_lag}")

        # --- Store the last row for future prediction ---
        # This should be the row with the latest 'Date' *before* dropping NaNs caused by lagging
        # It contains the most recent actual values needed to predict the *next* day
        if not self.df.empty:
             self.last_original_row_for_prediction = self.df.iloc[-1:].copy()
             print(f"Stored the last row (Date: {self.last_original_row_for_prediction['Date'].iloc[0]}) for future prediction input.")
        else:
             raise ValueError("DataFrame is empty after cleaning/lagging prep.")


        # --- Handle Missing Values ---
        initial_rows = len(self.df)
        # Drop rows with NaN values resulting from lagged features or initial NaNs/conversion errors
        self.df.dropna(inplace=True)
        rows_after_dropna = len(self.df)

        if self.df.empty:
            raise ValueError("DataFrame is empty after dropping NaN values. Check lagging or initial data quality.")


        # --- Define Features (X) and Target (y) ---
        # Target variable 'Price' must exist
        if 'Price' not in self.df.columns:
             raise ValueError("Target column 'Price' is not available after preprocessing.")
        self.y = self.df['Price']

        # Features are all columns except 'Date' and 'Price'
        self.x = self.df.drop(['Date', 'Price'], axis=1)
        self.feature_names = self.x.columns.tolist() # Store feature names

        # Final check for NaNs in features or target
        if self.x.isnull().values.any():
            print("Warning: NaNs detected in features (X) after final processing.")
            print(self.x[self.x.isnull().any(axis=1)]) # Show rows with NaNs
            # Optionally, handle again (e.g., imputation) or raise error
            # self.x.fillna(self.x.median(), inplace=True) # Example: Impute with median
            raise ValueError("NaNs found in features (X) before scaling. Cannot proceed.")
        if self.y.isnull().values.any():
            print("Warning: NaNs detected in target (y) after final processing.")
            raise ValueError("NaNs found in target (y). Cannot proceed.")

        print(f"Preprocessing complete. X shape: {self.x.shape}, y shape: {self.y.shape}")

        return self.x, self.y


    def scale_features(self):
        if self.x is None:
            raise ValueError("Features (X) not available. Run preprocess_data first.")
        self.scaler = StandardScaler()
        # Fit on the entire feature set X before splitting
        self.x_scaled = self.scaler.fit_transform(self.x)
        print("Feature scaling complete using StandardScaler.")
        # Return the scaled features as a numpy array
        return self.x_scaled

    def get_last_features_for_prediction(self):
        """
        Prepares the most recent feature set needed to predict the next day.
        Uses the stored 'last_original_row_for_prediction'.
        """
        if self.last_original_row_for_prediction is None:
            raise ValueError("Last original row for prediction not stored. Run preprocess_data.")
        if self.scaler is None:
             raise ValueError("Scaler not fitted. Run scale_features first.")
        if self.feature_names is None:
             raise ValueError("Feature names not stored. Run preprocess_data first.")

        # Construct the feature vector for the *next* prediction step.
        # This requires shifting the lagged features from the last known row.
        # Example: To predict Day T+1, we need Price_lag_1 (which is Price at T),
        # Price_lag_2 (which is Price at T-1), ..., Vol_lag_1 (Vol at T), etc.

        # Select only the feature columns from the last stored row
        last_features_raw = self.last_original_row_for_prediction[self.feature_names].copy()

        # Check for NaNs in the raw features selected
        if last_features_raw.isnull().values.any():
             print("Warning: NaNs found in the last raw feature row selected for prediction:")
             print(last_features_raw[last_features_raw.isnull().any(axis=1)])
             # Attempt imputation (e.g., using the mean/median from the training data's corresponding column)
             # This is tricky as the scaler was fit on the non-NaN data.
             # A simpler approach might be to ensure the very last row used has no NaNs before storing it.
             # For now, raising an error is safer.
             raise ValueError("NaNs found in the feature set prepared for future prediction. Cannot proceed.")


        # Scale these features using the *already fitted* scaler
        last_features_scaled = self.scaler.transform(last_features_raw) # Use transform, not fit_transform

        print(f"Prepared and scaled features for next-day prediction. Shape: {last_features_scaled.shape}")
        return last_features_scaled

    def split_data(self, test_size = 0.10, random_state=42, shuffle=True): # Changed shuffle to False for time series
        # Use the scaled features (self.x_scaled)
        if self.x_scaled is None:
             raise ValueError("Scaled features not available. Run scale_features first.")

        # Important for time series: DO NOT shuffle the data when splitting.
        # The order matters, test set should come after train set.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_scaled, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        print(f"Data split complete.\n")
        return self.x_train, self.x_test, self.y_train, self.y_test




#Base class for machine learning models
class Model_(Preprocessing):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.model = None # Placeholder for the actual model instance
        self.history = None # For storing training history (e.g., from Keras)
        self.next_day_prediction = None # Store the future prediction result

        # --- Core Preprocessing Steps ---
        self.x, self.y = self.preprocess_data()
        self.x_scaled = self.scale_features() # This now returns the scaled data
        # Prepare the feature vector for predicting the day *after* the dataset ends
        self.last_scaled_features = self.get_last_features_for_prediction()

        # --- Splitting Data ---
        # Use the scaled data for splitting. Crucially, turn OFF shuffling for time series.
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(shuffle=True)

        # --- Ensure y_train and y_test are float ---
        self.y_train = self.y_train.astype(float)
        self.y_test = self.y_test.astype(float)

        # --- RNN Specific Reshaping (Done in RNN class __init__) ---
        # Moved reshaping logic to the RNN class itself to avoid errors
        # when other models are selected.

    def train(self):
        """Placeholder for training logic. Implemented by subclasses."""
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def evaluate(self):
        """Evaluates the model on the test set and saves metrics and plots."""
        if self.model is None:
             print("Error: Model has not been trained or loaded.")
             return None

        output_dir = self.get_output_dir()
        print(f"\n--- Evaluating {self.__class__.__name__} ---")
        print(f"Using test data: X_test shape {self.x_test.shape}, y_test shape {self.y_test.shape}")

        # --- Predict on Test Set ---
        try:
            # The predict method should handle potential reshaping needed (like for RNN)
            y_pred = self.predict(self.x_test)
            if y_pred is None:
                 print("Prediction on test set failed.")
                 return None
            # Ensure y_pred is a flat array for metrics calculation
            if y_pred.ndim > 1 and y_pred.shape[1] == 1:
                 y_pred = y_pred.flatten()

        except Exception as e:
            print(f"Error during prediction on test set: {e}")
            traceback.print_exc()
            return None

        # --- Calculate Regression Metrics ---
        try:
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            print(f"Evaluation Metrics:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R2 Score: {r2:.4f}")

            metrics_contents = (f'Type: Regression\n'
                                f'Mean Squared Error (MSE): {mse:.4f}\n'
                                f'Root Mean Squared Error (RMSE): {rmse:.4f}\n'
                                f'Mean Absolute Error (MAE): {mae:.4f}\n'
                                f'R-squared (R2): {r2:.4f}\n')
            metrics_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

            # Save metrics to file
            metrics_path = os.path.join(output_dir, "metrics.txt")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as file:
                file.write(metrics_contents)
            print(f'Metrics saved to {metrics_path}')

        except Exception as e:
            print(f"Error calculating or saving metrics: {e}")
            traceback.print_exc()
            metrics_dict = None # Indicate failure

        # --- Plot Actual vs Predicted ---
        try:
            plt.figure(figsize=(7, 6)) # Slightly larger figure
            plt.scatter(self.y_test, y_pred, alpha=0.6, edgecolors='k', s=50) # Added edgecolors and size
            # Plot the ideal fit line based on actual data range
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], '--r', lw=2, label='Ideal Fit (y=x)')
            plt.xlabel('Actual Price ($)', fontsize=12)
            plt.ylabel('Predicted Price ($)', fontsize=12)
            plt.title(f'{self.__class__.__name__}: Actual vs. Predicted Prices', fontsize=14)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout() # Adjust layout
            plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved Actual vs Predicted plot to: {plot_path}")
        except Exception as e:
            print(f"Error generating or saving Actual vs Predicted plot: {e}")
            traceback.print_exc()

        # --- Predict Future Day ---
        try:
             print("\n--- Predicting Next Day's Price ---")
             # Use the predict_future method which should be implemented by subclasses
             self.next_day_prediction = self.predict_future(n_days=1)
             if self.next_day_prediction is not None:
                  # Ensure it's a single value if prediction was successful
                  if isinstance(self.next_day_prediction, (np.ndarray, list)):
                       self.next_day_prediction = self.next_day_prediction[0] # Take the first element

                  print(f"Predicted price for the next day: ${self.next_day_prediction:,.2f}")

                  # Optionally save this prediction to a file
                  pred_path = os.path.join(output_dir, "next_day_prediction.txt")
                  with open(pred_path, 'w') as f:
                       f.write(f"Predicted Price for Next Day: {self.next_day_prediction:.4f}\n")
                  print(f"Next day prediction saved to: {pred_path}")

             else:
                  print("Failed to generate next day prediction.")

        except Exception as e:
             print(f"Error during future prediction: {e}")
             traceback.print_exc()
             self.next_day_prediction = None


        # --- Save Learning Curves (if applicable, e.g., for RNN) ---
        if self.history is not None:
            print("\n--- Saving Learning Curves ---")
            self._save_learning_curves(self.history)

        return metrics_dict # Return metrics calculated on the test set


    def predict(self, X):
        """Placeholder for prediction logic on given data X. Implemented by subclasses."""
        raise NotImplementedError("Predict method must be implemented by subclasses.")

    def predict_future(self, n_days=1):
        """
        Predicts the price for n_days into the future using the last available data.
        Must be implemented by subclasses.
        Requires self.last_scaled_features to be prepared in __init__.
        """
        raise NotImplementedError("predict_future method must be implemented by subclasses.")


    def get_output_dir(self):
        """Gets the algorithm-specific output directory."""
        base_output_dir = "all_outputs"
        algo_name = self.__class__.__name__
        specific_dir = os.path.join(base_output_dir, f"{algo_name.replace(' ', '')}_output")
        os.makedirs(specific_dir, exist_ok=True) # Create dir if it doesn't exist
        return specific_dir

    def _save_learning_curves(self, history):
        """Saves learning curves (e.g., loss, MAE) if history object is available."""
        output_dir = self.get_output_dir()

        if not hasattr(history, 'history') or not isinstance(history.history, dict):
            print("Warning: History object invalid or does not contain 'history' dictionary. Cannot save learning curves.")
            return

        history_dict = history.history
        possible_metrics = {
            'Loss': 'loss',
            'Mean Absolute Error': 'mae',
            # Add others if tracked, e.g., 'Mean Squared Error': 'mse'
        }

        print("Attempting to save learning curves...")
        plotted_any = False

        for display_name, key_base in possible_metrics.items():
            train_key = key_base
            val_key = f'val_{key_base}'

            if train_key in history_dict and val_key in history_dict:
                plotted_any = True
                try:
                    epochs = range(1, len(history_dict[train_key]) + 1)
                    plt.figure(figsize=(8, 5)) # Standard figure size
                    plt.plot(epochs, history_dict[train_key], 'bo-', label=f'Training {display_name}', markersize=4) # Added markers
                    plt.plot(epochs, history_dict[val_key], 'rs-', label=f'Validation {display_name}', markersize=4) # Added markers
                    plt.title(f'{self.__class__.__name__}: Model {display_name} Over Epochs', fontsize=14)
                    plt.ylabel(display_name, fontsize=12)
                    plt.xlabel('Epoch', fontsize=12)
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    file_name = f"learning_curve_{key_base}.png"
                    curve_path = os.path.join(output_dir, file_name)
                    plt.savefig(curve_path)
                    plt.close()
                    print(f"Saved {display_name} curve to: {curve_path}")
                except Exception as e:
                    print(f"Error saving {display_name} curve: {e}")
                    traceback.print_exc()
            else:
                print(f"Skipping plot for {display_name}: Keys '{train_key}' or '{val_key}' not found in history.")

        if not plotted_any:
            print("Warning: No suitable keys found in history object to plot any learning curves.")




#Random Forrest class
class RandomForest(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        print("RandomForestRegressor initialized.")

    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_future(self, n_days=1):
        if self.model is None:
             print("Error: RandomForest model is not trained.")
             return None
        if self.last_scaled_features is None:
             print("Error: Last scaled features for prediction are not available.")
             return None
        if n_days != 1:
            print("Warning: RandomForest future prediction currently only supports n_days=1.")
            # Add logic for multi-step prediction if needed later (more complex)

        try:
             # Predict using the last known scaled features
             prediction = self.model.predict(self.last_scaled_features)
             # Return the single predicted value (or first value if array)
             return prediction[0] if isinstance(prediction, np.ndarray) else prediction
        except Exception as e:
             print(f"Error during RandomForest future prediction: {e}")
             traceback.print_exc()
             return None


#SVM class
class SupportVectorMachine(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn import svm

        C = 1.0
        epsilon = 0.1

        self.model = svm.SVR(kernel = 'linear', C=C, epsilon=epsilon, cache_size=500)

    def train(self):
        print("Training Support Vector Regressor...")
        self.model.fit(self.x_train, self.y_train)
        print("SVR Training Complete.")

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def predict_future(self, n_days=1):
        if self.model is None:
             print("Error: SVR model is not trained.")
             return None
        if self.last_scaled_features is None:
             print("Error: Last scaled features for prediction are not available.")
             return None
        if n_days != 1:
            print("Warning: SVR future prediction currently only supports n_days=1.")

        try:
             # Predict using the last known scaled features
             prediction = self.model.predict(self.last_scaled_features)
             return prediction[0] if isinstance(prediction, np.ndarray) else prediction
        except Exception as e:
             print(f"Error during SVR future prediction: {e}")
             traceback.print_exc()
             return None


# Linear Regression class
class LinearRegression(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.linear_model import LinearRegression  # Import Linear Regression model
        
        self.model = LinearRegression()  # Initialize the model
    
    def train(self):
        self.model.fit(self.x_train, self.y_train)  # Train the linear regression model
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def predict_future(self, n_days=1):
        if self.model is None:
             print("Error: LinearRegression model is not trained.")
             return None
        if self.last_scaled_features is None:
             print("Error: Last scaled features for prediction are not available.")
             return None
        if n_days != 1:
            print("Warning: LinearRegression future prediction currently only supports n_days=1.")

        try:
             # Predict using the last known scaled features
             prediction = self.model.predict(self.last_scaled_features)
             return prediction[0] if isinstance(prediction, np.ndarray) else prediction
        except Exception as e:
             print(f"Error during LinearRegression future prediction: {e}")
             traceback.print_exc()
             return None


#RNN class
class RNN(Model_):
    def __init__(self, dataset_path):
        if not tf_imported or not Sequential: # Check necessary components
             raise ImportError("TensorFlow components required for RNN are not available. Cannot initialize RNN model.")


        super().__init__(dataset_path)
        #from tensorflow.keras.models import Sequential
        #from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
        
        
        # --- Reshape data specifically for RNN ---
        # Reshape feature data for RNN [samples, time steps, features]
        # Important: Reshape train and test sets *after* they are created by parent __init__
        # Assuming 1 time step per sample based on your original logic
        try:
            self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
            self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))
            print(f"RNN data reshaped. x_train shape: {self.x_train.shape}, x_test shape: {self.x_test.shape}")
        except Exception as e:
             print(f"Error reshaping data for RNN: {e}")
             traceback.print_exc()
             raise ValueError("Failed to reshape data for RNN input.")


        # --- Define the RNN Model Architecture ---
        self.model = Sequential(name="Bitcoin_RNN") # Give the model a name
        # Input shape: (time_steps, num_features) -> (1, num_features)
        input_shape = (self.x_train.shape[1], self.x_train.shape[2])

        # Add layers
        # Consider using LSTM or GRU for potentially better performance on time series
        self.model.add(SimpleRNN(units=50, return_sequences=True, input_shape=input_shape)) # Keep sequences for next RNN layer
        self.model.add(Dropout(0.2))
        self.model.add(SimpleRNN(units=50, return_sequences=False)) # Last RNN layer returns single output
        self.model.add(Dropout(0.2))
        # self.model.add(Dense(units=25, activation='relu')) # Reduced dense layer size
        self.model.add(Dense(units=1, activation='linear')) # Output layer for regression

        # Compile the model
        # Use Mean Squared Error for regression. Adam optimizer is common. Track MAE.
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print("RNN Model Architecture:")
        self.model.summary() # Print model summary to console


    def train(self):

        #import tensorflow
        #from tensorflow.keras.callbacks import ModelCheckpoint
        if self.model is None:
             raise ValueError("RNN Model not initialized.")
        if not tf_imported or not ModelCheckpoint: # Check again before training
             print("Error: TensorFlow components not available. Cannot train RNN.")
             return

        output_dir = self.get_output_dir()
        checkpoint_filepath = os.path.join(output_dir, 'rnn_best_weights.weights.h5')
        #model_save_filepath = os.path.join(output_dir, 'rnn_entire_model.h5')

        # --- Create the ModelCheckpoint callback ---
        # Monitor 'val_loss' to save the model when validation loss is lowest
        # save_weights_only=True saves only the weights
        # save_best_only=True ensures only the best model (lowest val_loss) is kept

        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss', # Or 'val_accuracy' if preferred
            mode='min',         # 'min' for loss, 'max' for accuracy
            save_best_only=True,
            verbose=1           # Print messages when weights are saved
        )

        nb_epochs = 25
        batch_size = 32

        print(f"\n--- Starting RNN Training (saving best weights to {checkpoint_filepath}) ---")
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=nb_epochs,
            batch_size=batch_size,
            validation_data=(self.x_test, self.y_test),
            verbose=2,
            callbacks=[cp_callback]
        )
        print("--- RNN Training Finished ---")


        # --- Load Best Weights ---
        # After training, load the weights from the epoch with the best validation loss
        if os.path.exists(checkpoint_filepath):
                print(f"Loading best weights from {checkpoint_filepath}...")
                self.model.load_weights(checkpoint_filepath)
                print("Best weights loaded into the model.")
        else:
                print("Warning: Best weights checkpoint file not found. Using weights from the last epoch.")

        '''
        try:
            self.model.save(model_save_filepath)
            print(f"Entire RNN model saved to: {model_save_filepath}")
        except Exception as e:
            print(f"Error saving the entire RNN model: {e}")
        '''

        self._save_learning_curves(self.history)


    def predict(self, X):
        """Predicts using the trained RNN model. Handles reshaping if necessary."""
        if self.model is None:
             print("Error: RNN model is not trained or loaded.")
             return None

        try:
            # Ensure input X has the correct 3D shape [samples, time_steps, features]
            if X.ndim == 2:
                 # Assume time_steps=1 if input is 2D
                 X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))
                 print(f"Input X reshaped for RNN prediction to: {X_reshaped.shape}")
            elif X.ndim == 3 and X.shape[1] == 1:
                 X_reshaped = X # Already in correct shape
            else:
                 print(f"Error: Input X has unexpected shape {X.shape} for RNN prediction.")
                 return None

            predictions = self.model.predict(X_reshaped)
            # Return predictions (usually a 2D array [samples, 1], flatten if needed)
            return predictions # Keep as is, evaluate handles flattening if needed
        except Exception as e:
             print(f"Error during RNN prediction: {e}")
             traceback.print_exc()
             return None

    def predict_future(self, n_days=1):
        """Predicts the next day's price using the last available data point."""
        if self.model is None:
             print("Error: RNN model is not trained or loaded.")
             return None
        if self.last_scaled_features is None:
             print("Error: Last scaled features for prediction are not available.")
             return None
        if n_days != 1:
            print("Warning: RNN future prediction currently only supports n_days=1.")
            # Multi-step prediction requires iterative prediction or a different model structure

        try:
            # --- Reshape the last scaled features for RNN input ---
            # Shape should be [1, time_steps, features] -> [1, 1, num_features]
            input_data = np.reshape(self.last_scaled_features, (1, 1, self.last_scaled_features.shape[1]))
            print(f"Input data for future prediction reshaped to: {input_data.shape}")

            # --- Predict using the model ---
            prediction = self.model.predict(input_data)

            # The prediction is likely a numpy array like [[value]], extract the scalar value
            future_price = prediction[0, 0]

            return future_price
        except Exception as e:
            print(f"Error during RNN future prediction: {e}")
            traceback.print_exc()
            return None

