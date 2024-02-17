from tensorflow.keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.sequence import pad_sequences

# Define paths to your data files
data_directory = (r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\TensorFlow\TrainingVideosLipsOnly')
data_files = os.listdir(data_directory)

# Initialize lists to store your training data and labels
X_train = []
y_train = []

# Process each data file
for file_name in data_files:
    file_path = os.path.join(data_directory, file_name)
    coordinates = process_data_file(file_path)
    # Append coordinates to X_train
    X_train.append(coordinates)
    # Extract label from the file name
    label = file_name.split('.')[0]  # Extract "yes" or "no" from the file name
    y_train.append(label)

# Pad sequences to a fixed length
X_train = pad_sequences(X_train, dtype='float32', padding='post', value=0.0)

# Convert lists to numpy arrays
y_train = np.array(y_train)

# Convert labels to binary format (0 for "no", 1 for "yes")
y_train = np.where(y_train == 'yes', 1, 0)

# Build, compile, and train your RNN model using the provided code
# Assuming you have the code provided in your question here

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)