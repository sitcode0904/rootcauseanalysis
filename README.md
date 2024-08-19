# Root Cause Analysis with Neural Networks
This project leverages deep learning to predict the root cause of system errors and issues based on historical data. The model is built using TensorFlow and Keras, and it utilizes a neural network architecture to analyze patterns and improve the accuracy of predictions.

Table of Contents
Dataset
Installation
Model Architecture
Training and Evaluation
Usage
Results
Contributing
License
Dataset
The dataset contains system metrics and corresponding root causes for various issues. The columns in the dataset are as follows:

ID: Identifier for each entry
CPU_LOAD: Integer values representing CPU load
MEMORY_LEAK_LOAD: Integer values representing memory leak load
DELAY: Integer values representing delay
ERROR_1000 to ERROR_1003: Integer values representing different types of errors
ROOT_CAUSE: The target variable, indicating the root cause of the issue
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/root-cause-analysis.git
Navigate to the project directory:
bash
Copy code
cd root-cause-analysis
Install the required Python packages:
bash
Copy code
pip install -r requirements.txt
Model Architecture
The neural network is designed with the following architecture:

Input Layer: Accepts 7 features (CPU_LOAD, MEMORY_LEAK_LOAD, DELAY, ERROR_1000, ERROR_1001, ERROR_1002, ERROR_1003).
Hidden Layers: Three hidden layers with ReLU activation and Dropout layers to prevent overfitting.
Output Layer: A softmax layer for multi-class classification, predicting the root cause.
Training and Evaluation
The model is trained for 10 epochs using the sparse_categorical_crossentropy loss function and the Adam optimizer. The dataset is split into training and test sets, with 80% used for training and 20% for testing.

Metrics
Accuracy: The accuracy of the model on the test set.
Root Mean Square Error (RMSE): Measures the difference between predicted and actual values.
Training and Validation Accuracy: Plotted to visualize model performance over epochs.
Usage
Load the dataset and preprocess the data.
Train the model using the provided code.
Evaluate the model on test data to assess performance.
Predict root causes for new data points using the trained model.
Example
python
Copy code
new_data = pd.DataFrame({
    'CPU_LOAD': [1],
    'MEMORY_LEAK_LOAD': [0],
    'DELAY': [1],
    'ERROR_1000': [0],
    'ERROR_1001': [0],
    'ERROR_1002': [1],
    'ERROR_1003': [0],
})

# Standardize and predict
new_data_scaled = scaler.transform(new_data)
new_prediction = model.predict(new_data_scaled)
predicted_class = label_encoder.inverse_transform(np.argmax(new_prediction, axis=1))

print(f'Prediction for new values: {predicted_class[0]}')
Results
Test Accuracy: The model achieved an accuracy of approximately xx.xx% on the test set.
RMSE: The Root Mean Square Error is x.xxx.
Performance metrics and training/validation plots are provided in the notebook or script for further analysis.

Contributing
Contributions are welcome! Please open an issue or submit a pull request with your improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.

