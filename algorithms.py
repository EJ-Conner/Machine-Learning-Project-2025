
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


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
        self.df = pd.read_csv(dataset_path)
        
    def preprocess_data(self):
        
        #remove %, K, M, and ',' to have numeric dataset 
        self.df['Change %'] = self.df['Change %'].str.replace('%', '').astype(float) / 100.0
        #use convert_volume function
        self.df['Vol.'] = self.df['Vol.'].apply(convert_volume)

        for col in ['Price', 'Open', 'High', 'Low']:
            self.df[col] = self.df[col].str.replace(',','').astype(float)

        #Create lags up to 5 days 
        # lagged features are past values of a variable
        # used as predictors for future variables
        
        for i in range(1, 6): 
            self.df[f'Price_lag_{i}'] = self.df['Price'].shift(i)
            self.df[f'Change_%_lag_{i}'] = self.df['Change %'].shift(i)
            self.df[f'Vol._lag_{i}'] = self.df['Vol.'].shift(i)
            self.df[f'High_lag_{i}'] = self.df['High'].shift(i)
            self.df[f'Low_lag_{i}'] = self.df['Low'].shift(i)
            self.df[f'Open_lag_{i}'] = self.df['Open'].shift(i)
    
        # Drop rows with NaN values resulting from lagged features
        self.df.dropna(inplace=True)
        
        #Split into features and target variable
        self.x = self.df.drop(['Date', 'Price'], axis=1) 
        self.y = self.df['Price']
        
        print(f"Preprocessing complete. X shape: {self.x.shape}, y shape: {self.y.shape}")
        if self.x.isnull().values.any() or self.y.isnull().values.any():
             print("Warning: NaNs detected in X or y after preprocessing and dropna.")
        
        return self.x, self.y

        
    def scale_features(self):
        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(self.x)
        print("Feature scaling complete.")
        return self.x

    def split_data(self, test_size = 0.25):
        #Split the dataset into training and test sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state=42)
        print("Data split complete.")
        return self.x_train, self.x_test, self.y_train, self.y_test

#Base class for machine learning models
class Model_(Preprocessing):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)

        #Preprocessing:
        self.x, self.y = self.preprocess_data()
        self.x = self.scale_features()

        self.x_train, self.x_test, self.y_train, self.y_test = self.split_data()

         # --- Ensure y_train and y_test are float (good practice) ---
        # It's safe to do this after split_data() for all models if needed
        self.y_train = self.y_train.astype(float)
        self.y_test = self.y_test.astype(float)

        # --- Apply RNN specific reshaping *after* splitting ---
        if self.__class__.__name__ == 'RNN':
            # Reshape feature data for RNN [samples, time steps, features]
            # Important: Reshape train and test sets separately
            # Assuming 1 time step per sample based on your original reshape logic
            self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
            self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))
            print(f"RNN data reshaped. x_train shape: {self.x_train.shape}, x_test shape: {self.x_test.shape}")
            # The y data (y_train, y_test) does not need reshaping for standard RNN regression output

        '''
        if self.__class__.__name__ == 'RNN':
            #Reshape data for RNN [samples, time steps, features]
            self.x = np.reshape(self.x, (self.x.shape[0], 1, self.x.shape[1]))
            self.y_train = self.y_train.astype(float)
            self.y_test = self.y_test.astype(float)
        '''
        #self.x_train, self.x_test, self.y_train, self.y_test = self.split_data()

    def train(self):
        pass

    #method to get the algorithm-specific output directory
    def get_output_dir(self):
        base_output_dir = "all_outputs"
        #Generate a folder name like 'RandomForest_output' from the class name
        algo_name = self.__class__.__name__
        specific_dir = os.path.join(base_output_dir, f"{algo_name.replace(' ', '')}_output")
        os.makedirs(specific_dir, exist_ok=True) # Create dir if it doesn't exist
        return specific_dir


    def evaluate(self):
        #Get the specific output directory for algorithm
        output_dir = self.get_output_dir()

        #Predicting on the test set
        y_pred, _ = self.predict(self.x_test)


        # ------REGRESSION EVALUATION-------
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # actual vs predicted
        try:
            plt.figure(figsize=(6, 5))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], '--r', lw=2, label='Ideal Fit')
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title('Actual vs. Predicted Values')
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(output_dir, "actual_vs_predicted.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved Actual vs Predicted plot to: {plot_path}")
        except Exception as e:
            print(f"Error saving Actual vs Predicted plot: {e}")

        
        metrics_contents = ((f'Type: Regression\n'
                            f'Mean Squared Error (MSE): {mse:.4f}\n'
                            f'Root Mean Squared Error (RMSE): {rmse:.4f}\n'
                            f'Mean Absolute Error (MAE): {mae:.4f}\n'
                            f'R-squared (R2): {r2:.4f}\n'))

        metrics_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

        try:
            metrics_path = os.path.join(output_dir, "metrics.txt")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as file:
                file.write(metrics_contents)
                print(f'File was successfully saved to {metrics_path}')
        except Exception as e:
            print(f"Error saving metrics.txt: {e}")
        

        return metrics_dict
        

    def _save_learning_curves(self, history):

        # Saves accuracy and loss learning curves using Matplotlib to the algorithm's specific folder.
        output_dir = self.get_output_dir() # Get the correct directory

        if not hasattr(history, 'history'):
            print("Error: History object does not contain 'history' attribute.")
            return

        history_dict = history.history
        possible_metrics = {
            'Loss': 'loss',
            'Mean Absolute Error': 'mae',
             # Add others if you track them, e.g. 'Mean Squared Error': 'mse'
             # 'Accuracy': 'accuracy' # Keep this commented out or remove for regression
        }

        print("Saving learning curves...")
        plotted_any = False

        for display_name, key_base in possible_metrics.items():
            train_key = key_base
            val_key = f'val_{key_base}'

            if train_key in history_dict and val_key in history_dict:
                plotted_any = True
                try:
                    plt.figure()
                    plt.plot(history_dict[train_key], label=f'Train {display_name}')
                    plt.plot(history_dict[val_key], label=f'Validation {display_name}')
                    plt.title(f'Model {display_name}')
                    plt.ylabel(display_name)
                    plt.xlabel('Epoch')
                    plt.legend()
                    plt.grid(True)
                    # Generate filename like 'learning_curve_loss.png', 'learning_curve_mean_absolute_error.png'
                    file_name = f"learning_curve_{key_base}.png"
                    curve_path = os.path.join(output_dir, file_name)
                    plt.savefig(curve_path)
                    plt.close()
                    print(f"Saved {display_name} curve to: {curve_path}")
                except Exception as e:
                    print(f"Error saving {display_name} curve: {e}")
            else:
                 print(f"Skipping plot for {display_name}: Keys '{train_key}' or '{val_key}' not found in history.")

        if not plotted_any:
             print("Warning: No suitable keys found in history object to plot any learning curves.")
        

    def predict(self, X):
        pass

#Random Forrest class
class RandomForest(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.ensemble import RandomForestRegressor

        self.rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    def train(self):
        self.rf.fit(self.x_train, self.y_train)

    def predict(self, X):
        return self.rf.predict(X), None

#SVM class
class SupportVectorMachine(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn import svm

        C = 1.0
        epsilon = 0.1

        self.svm = svm.SVR(kernel = 'rbf', C=C, epsilon=epsilon)

    def train(self):
        print("Training Support Vector Regressor...")
        self.svm.fit(self.x_train, self.y_train)
        print("SVR Training Complete.")

    def predict(self, X):
        predictions = self.svm.predict(X)
        return predictions, None


# Linear Regression class
class LinearRegression(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from sklearn.linear_model import LinearRegression  # Import Linear Regression model
        
        self.lr = LinearRegression()  # Initialize the model
    
    def train(self):
        self.lr.fit(self.x_train, self.y_train)  # Train the linear regression model
    
    def predict(self, X):
        return self.lr.predict(X), None  # Predict values for the input X


#RNN class
class RNN(Model_):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
        
        #Model
        self.model = Sequential()
        #RNN layers
        self.model.add(SimpleRNN(units=50, return_sequences=False, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=50, activation='relu'))
        self.model.add(Dense(units=1)) #, activation='linear'
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Use 'categorical_crossentropy' for multi-class
        self.model.summary()

    def train(self):

        import tensorflow
        from tensorflow.keras.callbacks import ModelCheckpoint

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

        nb_epochs = 10
        batch_size = 64

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

        '''
        try:
            self.model.save(model_save_filepath)
            print(f"Entire RNN model saved to: {model_save_filepath}")
        except Exception as e:
            print(f"Error saving the entire RNN model: {e}")
        '''

        self._save_learning_curves(self.history)


    def predict(self, X):
        y_pred = self.model.predict(self.x_test)
        return y_pred, y_pred

