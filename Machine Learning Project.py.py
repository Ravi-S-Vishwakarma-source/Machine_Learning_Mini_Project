#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load the dataset
#link for the dataset: https://www.kaggle.com/uciml/pima-indians-diabetes-database
file_path = r"C:\Users\vishw\OneDrive\Documents\Machine Learning\diabetes.csv" # Change the path to your file location
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Step 2: Split the data into features (X) and target (y)
X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]  # Target variable

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Standardize features (important for distance-based models like k-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Baseline Model (k-NN)
print("\n--- Baseline Model: k-NN Classifier ---")
knn = KNeighborsClassifier(n_neighbors=5)  # Default k=5
knn.fit(X_train_scaled, y_train)

# Evaluate k-NN
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
roc_auc_knn = roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:, 1])

print(f"k-NN Accuracy: {accuracy_knn:.4f}")
print(f"k-NN ROC-AUC Score: {roc_auc_knn:.4f}")
print("\nClassification Report for k-NN:")
print(classification_report(y_test, y_pred_knn))

# Step 6: CNN Model
print("\n--- CNN Model ---")

# Reshape data for CNN (add a channel dimension)
X_train_cnn = np.expand_dims(X_train_scaled, axis=-1)
X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)

# Build CNN model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_cnn, y_train, validation_split=0.2, epochs=100, batch_size=32,
                    callbacks=[early_stopping], verbose=1)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nCNN Test Loss: {test_loss:.4f}, CNN Test Accuracy: {test_accuracy:.4f}")

# Predict with CNN
y_pred_prob_cnn = model.predict(X_test_cnn).flatten()
y_pred_cnn = (y_pred_prob_cnn > 0.5).astype(int)

# CNN Metrics
roc_auc_cnn = roc_auc_score(y_test, y_pred_prob_cnn)
print("\nClassification Report for CNN:")
print(classification_report(y_test, y_pred_cnn))
print(f"CNN ROC-AUC Score: {roc_auc_cnn:.4f}")

# Step 7: Compare Results
print("\n--- Model Comparison ---")
print(f"k-NN Accuracy: {accuracy_knn:.4f}, k-NN ROC-AUC: {roc_auc_knn:.4f}")
print(f"CNN Accuracy: {test_accuracy:.4f}, CNN ROC-AUC: {roc_auc_cnn:.4f}")

# Conclusion
if test_accuracy > accuracy_knn:
    print("The CNN model outperforms the k-NN baseline.")
else:
    print("The k-NN baseline outperforms the CNN model.")

# ---- Graphs for Metrics ----

# Calculate the metrics for plotting
accuracy = test_accuracy  # CNN Accuracy
precision = precision_score(y_test, y_pred_cnn)
recall = recall_score(y_test, y_pred_cnn)
f1 = f1_score(y_test, y_pred_cnn)
roc_auc = roc_auc_cnn

# Plotting Accuracy, Precision, Recall, F1-Score in a bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
sns.barplot(x=metrics, y=values, palette='Blues_d')
plt.title('Model Performance Metrics for CNN')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.show()

# Plotting ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_cnn)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line representing random guess
plt.title('Receiver Operating Characteristic (ROC) Curve for CNN')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Generate and plot correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title('Correlation Heatmap for Pima Indians Diabetes Database')
plt.show()


# Print ROC-AUC Score
print(f"ROC-AUC Score: {roc_auc:.4f}")
