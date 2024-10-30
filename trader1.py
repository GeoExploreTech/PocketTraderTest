import os


import numpy as np
import pandas as pd
import random
import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Load the data (replace 'data.csv' with your data file path)
data = pd.read_csv("history-AUDNZD_otc.csv")

# Calculate the tail and head of the candle
data["tail"] = data[["open", "close"]].min(axis=1) - data["low"]
data["head"] = data["high"] - data[["open", "close"]].max(axis=1)


# Preprocess the data
scaler = MinMaxScaler()
data[["open", "high", "low", "close", "tail", "head"]] = scaler.fit_transform(
    data[["open", "high", "low", "close", "tail", "head"]]
)


# Parameters
initial_seq_length = 30  # initial sequence length
update_seq_length = 10  # sequence length for new data batches
epochs_per_batch = 5  # number of epochs for each new data batch


# Function to create sequences and labels
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(
            data[["open", "high", "low", "close", "tail", "head"]]
            .iloc[i : i + seq_length]
            .values
        )
        y.append(
            1
            if data["close"].iloc[i + seq_length]
            > data["close"].iloc[i + seq_length - 1]
            else 0
        )
    return np.array(X), np.array(y)


seq_length = 30  # length of the sequence
X, y = create_sequences(data, seq_length)

print(X)
print(y)
# Split the data
split_ratio = 0.8
split = int(len(X) * split_ratio)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model architecture
model = Sequential(
    [
        LSTM(
            50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])
        ),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# # Train the model on initial data
# model.fit(X_train, y_train, epochs=epochs_per_batch, batch_size=32, verbose=1)

# # Simulating new data arriving in batches
# new_data_start_idx = len(initial_data)

# # Loop to update model with each new batch of data
# while new_data_start_idx + update_seq_length < len(data):
#     # Get the next batch of new data
#     new_data = data[new_data_start_idx : new_data_start_idx + update_seq_length]

#     # Update the training data by appending new sequences
#     X_new, y_new = create_sequences(new_data, update_seq_length)
#     X_train = np.concatenate((X_train, X_new))
#     y_train = np.concatenate((y_train, y_new))

#     # Retrain model on the updated data
#     model.fit(X_train, y_train, epochs=epochs_per_batch, batch_size=32, verbose=1)

#     # Move index forward to simulate arrival of the next data batch
#     new_data_start_idx += update_seq_length

# # Test on remaining data
# X_test, y_test = create_sequences(data[new_data_start_idx:], initial_seq_length)
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Final Test Accuracy: {accuracy}")
