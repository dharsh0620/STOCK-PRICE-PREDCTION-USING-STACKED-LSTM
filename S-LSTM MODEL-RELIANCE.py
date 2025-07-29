import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Load the data from Excel file
file_path = r"C:\Users\rando\OneDrive\Desktop\IMPORTANT\STOCK-PRICE PREDCTION\STAND ALONE-S-LSTM\Reliance.xlsx"
df = pd.read_excel(file_path)

# Sort data by Date in ascending order (oldest to newest)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Assuming the relevant column is named 'Price' for stock prices (change if necessary)
price_data = df['Price'].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data)

# Prepare the dataset for training (using the last 'n' days for prediction)
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create the dataset with a time_step (look-back period)
time_step = 60  # Using last 60 days to predict next day's price
X, y = create_dataset(scaled_data, time_step)

# Reshape input for LSTM model (samples, time_step, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the dataset into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the Stacked LSTM model
def build_stacked_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the Stacked LSTM model and store training history
stacked_lstm_model = build_stacked_lstm_model((X_train.shape[1], 1))
history = stacked_lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions with the Stacked LSTM model
stacked_lstm_train_pred = stacked_lstm_model.predict(X_train)
stacked_lstm_test_pred = stacked_lstm_model.predict(X_test)

# Inverse transform predictions and actual values
stacked_lstm_train_pred = scaler.inverse_transform(stacked_lstm_train_pred)
stacked_lstm_test_pred = scaler.inverse_transform(stacked_lstm_test_pred)

y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions
plt.figure(figsize=(14, 8))

# Plot the training predictions (aligned with training dates)
train_dates = df['Date'][:train_size]
plt.plot(train_dates, y_train_actual, color='blue', label='Actual Price (Train)')
plt.plot(train_dates, stacked_lstm_train_pred, color='green', label='Stacked LSTM Prediction (Train)')

# Plot the test predictions (aligned with test dates)
test_dates = df['Date'][train_size:train_size + len(y_test_actual)]
plt.plot(test_dates, y_test_actual, color='red', label='Actual Price (Test)')
plt.plot(test_dates, stacked_lstm_test_pred, color='green', linestyle='--', label='Stacked LSTM Prediction (Test)')

# Mark the next predicted value as yellow dot (next day after the test data ends)
next_day_pred = stacked_lstm_model.predict(X_test[-1].reshape(1, time_step, 1))
next_day_pred = scaler.inverse_transform(next_day_pred)

# The next day prediction will be plotted on the last test date
next_day_date = test_dates.iloc[-1]  # Last date of test data
plt.scatter(next_day_date, next_day_pred, color='yellow', label='Next Day Prediction', zorder=5)

plt.title('Stock Price Prediction (RELIANCE) - Stacked LSTM')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates on x-axis for better readability
plt.grid(True)
plt.show()

# Evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = 100 - mape * 100  # Approx directional accuracy
    print(f"{model_name} => MSE: {mse:.4f}, MAPE: {mape:.4f}, Accuracy: {accuracy:.2f}%")

evaluate_model(y_train_actual, stacked_lstm_train_pred, "Stacked LSTM Train")
evaluate_model(y_test_actual, stacked_lstm_test_pred, "Stacked LSTM Test")

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
