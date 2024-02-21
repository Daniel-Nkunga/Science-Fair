import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

# Step 1: Load the data from CSV files
data_path = (r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\All')
words = ['yes', 'no', 'nice', 'yell']
num_samples = 50
data = []
labels = []

for word in words:
    for i in range(1, num_samples + 1):
        filename = f'{data_path}{word}{i}.csv'
        if os.path.exists(filename):  # Check if the file exists
            try:
                df = pd.read_csv(filename)
                if not df.empty:  # Check if dataframe is not empty
                    coordinates = df.values.flatten()
                    data.append(coordinates)
                    if word == 'yes':
                        labels.append(0)
                    elif word == 'no':
                        labels.append(1)
                    else:
                        labels.append(2)
                else:
                    print(f"Empty dataframe found in file '{filename}'. Skipping...")
            except FileNotFoundError:
                print(f"File '{filename}' not found. Skipping...")
        else:
            print(f"File '{filename}' not found. Skipping...")

# Check if data is not empty
if data:
    # Step 2: Preprocess the data
    scaler = StandardScaler()

    # Filter out columns with zero variance
    non_zero_variance_indices = np.var(data, axis=0) != 0
    data_with_non_zero_variance = np.array(data)[:, non_zero_variance_indices]

    if data_with_non_zero_variance.any():
        data = scaler.fit_transform(data_with_non_zero_variance)  # Normalize the data
    else:
        print("All columns have zero variance. Unable to normalize the data.")

    # Step 3: Organize the data
    X = np.array(data)
    y = np.array(labels)

    # Step 4: Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Define the CNN model
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # Output layer with 3 units for tertiary classification
    ])

    # Step 6: Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Step 7: Reshape input data for CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Step 8: Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Step 9: Evaluate the model on the validation set
    _, accuracy = model.evaluate(X_val, y_val)
    print("Accuracy:", accuracy)
else:
    print("No data available for processing.")
