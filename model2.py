import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import matplotlib.pyplot as plt

# Load and merge datasets
def load_and_validate_data():
    datasets = [
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-eye-region.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_hog_eye"),
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-whole-face.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_hog_face"),
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/OF-eye-region.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_of_eye"),
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/OF-whole-face.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_of_face"),
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/GEO-eye-region.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_geo_eye"),
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/GEO-whole-face.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_geo_face"),
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/LBP-eye-region.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_lb_eye"),
        (pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/LBP-whole-face.csv").drop(columns=["Unnamed: 0"], errors='ignore'), "_lb_face")
    ]

    data = datasets[0][0]  # Start with the first dataset
    for dataset, suffix in datasets[1:]:
        data = data.merge(dataset, on="image_id", suffixes=("", suffix))

    print("\nMerged data shape:", data.shape)
    print("\nMerged data columns:", data.columns)
    return data

# Prepare features and target variable
def prepare_features(data):
    print("\nClass distribution:")

    X = data.drop(columns=["image_id","autism_hog_face","autism_of_face","autism_geo_face","autism_lb_face","autism_geo_eye", "autism_lb_eye","autism"], errors='ignore')
    y = data["autism"]

    print("\nTop feature correlations with target:")
    correlations = X.corrwith(y).sort_values(ascending=False)
    print(correlations.head())
    print(correlations.tail())

    constant_features = X.columns[X.std() < 0.01]
    if constant_features.any():
        print("\nWarning: Found near-constant features:", constant_features)
        X = X.drop(columns=constant_features)

    corr_matrix = X.corr().abs()
    high_corr_pairs = np.where(np.triu(corr_matrix, 1) > 0.95)
    if len(high_corr_pairs[0]) > 0:
        print("\nWarning: Found highly correlated feature pairs:")
        for i, j in zip(*high_corr_pairs):
            print(f"{X.columns[i]} - {X.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

    return X, y

# Train-test split and standardization
def prepare_data_splits(X, y):
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_holdout_scaled = scaler.transform(X_holdout)

    print("\nData split sizes:")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    print(f"Holdout set: {X_holdout_scaled.shape}")

    return X_train_scaled, X_val_scaled, X_holdout_scaled, y_train, y_val, y_holdout

# Baseline models
def train_evaluate_baseline_models(X_train_scaled, X_val_scaled, X_holdout_scaled, y_train, y_val, y_holdout):
    lr = LogisticRegression(max_iter=1000)
    lr_cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
    print("\nLogistic Regression CV Scores:", lr_cv_scores)
    print("CV Average:", lr_cv_scores.mean())

    lr.fit(X_train_scaled, y_train)
    y_val_pred_lr = lr.predict(X_val_scaled)
    y_holdout_pred_lr = lr.predict(X_holdout_scaled)

    print("\nLogistic Regression Validation Set:")
    print(classification_report(y_val, y_val_pred_lr))
    print("\nLogistic Regression Holdout Set:")
    print(classification_report(y_holdout, y_holdout_pred_lr))

    svm = SVC(kernel="rbf", probability=True)
    svm_cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    print("\nSVM CV Scores:", svm_cv_scores)
    print("CV Average:", svm_cv_scores.mean())

    svm.fit(X_train_scaled, y_train)
    y_val_pred_svm = svm.predict(X_val_scaled)
    y_holdout_pred_svm = svm.predict(X_holdout_scaled)

    print("\nSVM Validation Set:")
    print(classification_report(y_val, y_val_pred_svm))
    print("\nSVM Holdout Set:")
    print(classification_report(y_holdout, y_holdout_pred_svm))

# Teacher Model
class TeacherModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout_rate=0.3):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.dropout1(self.bn1(torch.relu(self.fc1(x))))
        out = self.dropout2(self.bn2(torch.relu(self.fc2(out))))
        out = self.fc3(out)
        return out

# Student Model
class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout_rate=0.2):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.dropout(self.bn1(torch.relu(self.fc1(x))))
        out = self.fc2(out)
        return out

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_accuracy = correct_train / total_train
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

# Knowledge Distillation Training
def train_student_with_distillation(student_model, teacher_model, train_loader, val_loader,
                                   optimizer, T=2.0, alpha=0.5, num_epochs=20, patience=3):
    criterion_ce = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        student_model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            student_logits = student_model(batch_x)
            with torch.no_grad():
                teacher_logits = teacher_model(batch_x)
            
            # Calculate losses
            teacher_soft = torch.softmax(teacher_logits / T, dim=1)
            student_log_soft = torch.log_softmax(student_logits / T, dim=1)
            
            loss_ce = criterion_ce(student_logits, batch_y)
            loss_kl = nn.functional.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
            loss = alpha * loss_ce + (1 - alpha) * (T ** 2) * loss_kl
            
            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        # Validation phase
        student_model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = student_model(batch_x)
                val_loss += criterion_ce(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = correct_train / total_train
        val_accuracy = correct_val / total_val

        print(f"Student Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = student_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                student_model.load_state_dict(best_model_state)
                break

# Main execution
if __name__ == "__main__":
    data = load_and_validate_data()
    X, y = prepare_features(data)
    X_train_scaled, X_val_scaled, X_holdout_scaled, y_train, y_val, y_holdout = prepare_data_splits(X, y)

    train_evaluate_baseline_models(X_train_scaled, X_val_scaled, X_holdout_scaled, y_train, y_val, y_holdout)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    X_holdout_tensor = torch.tensor(X_holdout_scaled, dtype=torch.float32)
    y_holdout_tensor = torch.tensor(y_holdout.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    input_dim = X_train_scaled.shape[1]
    teacher_model = TeacherModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=1e-4)

    print("\nTraining Teacher Model:")
    train_model(teacher_model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=3)

    student_model = StudentModel(input_dim)
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-4)

    print("\nTraining Student Model with Knowledge Distillation:")
    train_student_with_distillation(
        student_model, 
        teacher_model, 
        train_loader, 
        val_loader,
        optimizer_student,
        T=2.0,          
        alpha=0.5,      
        num_epochs=20,
        patience=3
    )