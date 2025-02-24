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

# Load and Merge OF Features with data validation
def load_and_validate_data():
    hog_eye = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-eye-region.csv")
    hog_face = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-whole-face.csv")
    # of_eye = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-eye-region.csv")
    # of_face = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-whole-face.csv")
    # G_eye = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-eye-region.csv")
    # of_face = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-whole-face.csv")
    # of_eye = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-eye-region.csv")
    # of_face = pd.read_csv("C:/Users/HP/Desktop/DATASET/AutFBio/HOG-whole-face.csv")
    
    # Print basic statistics about the raw data
    print("Eye region data shape:", hog_eye.shape)
    print("Face data shape:", hog_face.shape)
    print("\nMissing values in eye region:", hog_eye.isnull().sum().sum())
    print("Missing values in face data:", hog_face.isnull().sum().sum())
    
    # Check for duplicate images
    print("\nDuplicate images in eye region:", hog_eye['image_id'].duplicated().sum())
    print("Duplicate images in face data:", hog_face['image_id'].duplicated().sum())
    
    # Merge and clean data
    data = hog_eye.merge(hog_face, on="image_id", suffixes=("_eye", "_face"))
    print("\nMerged data shape:", data.shape)
    
    return data

# Prepare Features & Target Variable with additional checks
def prepare_features(data):
    # Check class distribution
    print("\nClass distribution:")
    print(data["autism_face"].value_counts(normalize=True))
    
    X = data.drop(columns=["image_id", "autism_face","autism_eye"], errors='ignore')
    y = data["autism_face"]
    
    # Feature correlation with target
    print("\nTop feature correlations with target:")
    correlations = X.corrwith(y).sort_values(ascending=False)
    print(correlations.head())
    print(correlations.tail())
    
    # Check for constant or near-constant features
    constant_features = X.columns[X.std() < 0.01]
    if len(constant_features) > 0:
        print("\nWarning: Found near-constant features:", constant_features)
        X = X.drop(columns=constant_features)
    
    # Check for highly correlated features
    corr_matrix = X.corr().abs()
    high_corr_pairs = np.where(np.triu(corr_matrix, 1) > 0.95)
    if len(high_corr_pairs[0]) > 0:
        print("\nWarning: Found highly correlated feature pairs:")
        for i, j in zip(*high_corr_pairs):
            print(f"{X.columns[i]} - {X.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
    
    return X, y

# Enhanced Train-Test Split & Standardization
def prepare_data_splits(X, y):
    # First split: Create a holdout set
    X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    
    # Second split: Create train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_holdout_scaled = scaler.transform(X_holdout)
    
    print("\nData split sizes:")
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Validation set: {X_val_scaled.shape}")
    print(f"Holdout set: {X_holdout_scaled.shape}")
    
    return X_train_scaled, X_val_scaled, X_holdout_scaled, y_train, y_val, y_holdout

# Step 4: Baseline Models with Cross-validation
def train_evaluate_baseline_models(X_train_scaled, X_val_scaled, X_holdout_scaled, 
                                 y_train, y_val, y_holdout):
    # Logistic Regression
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
    
    # SVM Classifier
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

#TeacherModel with dropout and batch normalization
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

#StudentModel with dropout
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

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
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
    
    return train_losses, val_losses

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
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Get student predictions
            student_logits = student_model(batch_x)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_logits = teacher_model(batch_x)
            
            # Compute soft targets
            teacher_soft = torch.softmax(teacher_logits / T, dim=1)
            student_log_soft = torch.log_softmax(student_logits / T, dim=1)
            
            # Compute losses
            loss_ce = criterion_ce(student_logits, batch_y)
            loss_kl = nn.functional.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
            loss = alpha * loss_ce + (1 - alpha) * (T * T) * loss_kl
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        student_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = student_model(batch_x)
                val_loss += criterion_ce(outputs, batch_y).item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Student Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = student_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                student_model.load_state_dict(best_model_state)
                break

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    data = load_and_validate_data()
    X, y = prepare_features(data)
    X_train_scaled, X_val_scaled, X_holdout_scaled, y_train, y_val, y_holdout = prepare_data_splits(X, y)
    
    # Train and evaluate baseline models
    train_evaluate_baseline_models(X_train_scaled, X_val_scaled, X_holdout_scaled,
                                 y_train, y_val, y_holdout)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    X_holdout_tensor = torch.tensor(X_holdout_scaled, dtype=torch.float32)
    y_holdout_tensor = torch.tensor(y_holdout.values, dtype=torch.long)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train teacher model
    input_dim = X_train_scaled.shape[1]
    teacher_model = TeacherModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("\nTraining Teacher Model:")
    train_losses, val_losses = train_model(
        teacher_model, train_loader, val_loader,
        criterion, optimizer, num_epochs=50, patience=3
    )
    
    # Initialize and train student model
    student_model = StudentModel(input_dim)
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Continuing from previous code...
    
    print("\nTraining Student Model with Knowledge Distillation:")
    train_student_with_distillation(
        student_model, teacher_model, train_loader, val_loader,
        optimizer_student, T=2.0, alpha=0.5, num_epochs=20, patience=3
    )
    
    # Evaluate all models on holdout set
    print("\n=== Final Evaluation on Holdout Set ===")
    
    # Teacher Model Evaluation
    teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = teacher_model(X_holdout_tensor)
        _, teacher_predicted = torch.max(teacher_outputs, 1)
        teacher_accuracy = (teacher_predicted == y_holdout_tensor).float().mean()
        print("\nTeacher Model Metrics:")
        print("Accuracy:", teacher_accuracy.item())
        print("\nClassification Report:")
        print(classification_report(y_holdout_tensor, teacher_predicted))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_holdout_tensor, teacher_predicted))
    
    # Student Model Evaluation
    student_model.eval()
    with torch.no_grad():
        student_outputs = student_model(X_holdout_tensor)
        _, student_predicted = torch.max(student_outputs, 1)
        student_accuracy = (student_predicted == y_holdout_tensor).float().mean()
        print("\nStudent Model Metrics:")
        print("Accuracy:", student_accuracy.item())
        print("\nClassification Report:")
        print(classification_report(y_holdout_tensor, student_predicted))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_holdout_tensor, student_predicted))
    
    # Compare predictions between Teacher and Student
    agreement = (teacher_predicted == student_predicted).float().mean()
    print("\nTeacher-Student Agreement Rate:", agreement.item())
    
    # Analyze mistakes
    disagreement_mask = teacher_predicted != student_predicted
    if disagreement_mask.any():
        print("\nAnalyzing disagreements between Teacher and Student models:")
        print("Number of disagreements:", disagreement_mask.sum().item())
        print("Disagreement indices:", torch.where(disagreement_mask)[0].tolist())
    
    # Model Size Comparison
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    print("\nModel Size Comparison:")
    print(f"Teacher Model Parameters: {count_parameters(teacher_model):,}")
    print(f"Student Model Parameters: {count_parameters(student_model):,}")
    
    # Save results to file
    results = {
        'data_stats': {
            'total_samples': len(data),
            'feature_count': X.shape[1],
            'class_distribution': y.value_counts().to_dict()
        },
        'model_performance': {
            'teacher': {
                'accuracy': teacher_accuracy.item(),
                'parameters': count_parameters(teacher_model)
            },
            'student': {
                'accuracy': student_accuracy.item(),
                'parameters': count_parameters(student_model)
            },
            'agreement_rate': agreement.item()
        }
    }
    
    # Save results to JSON file
    import json
    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults have been saved to 'model_evaluation_results.json'")
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Teacher confusion matrix
    teacher_cm = confusion_matrix(y_holdout_tensor, teacher_predicted)
    ax1.imshow(teacher_cm, cmap='Blues')
    ax1.set_title('Teacher Model\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Student confusion matrix
    student_cm = confusion_matrix(y_holdout_tensor, student_predicted)
    im = ax2.imshow(student_cm, cmap='Blues')
    ax2.set_title('Student Model\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # Add colorbar
    plt.colorbar(im, ax=(ax1, ax2))
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("\nConfusion matrices have been saved to 'confusion_matrices.png'")

# [Previous code remains the same until the main execution part]

def main():
    # Load and prepare data
    data = load_and_validate_data()
    X, y = prepare_features(data)
    X_train_scaled, X_val_scaled, X_holdout_scaled, y_train, y_val, y_holdout = prepare_data_splits(X, y)
    
    # Train and evaluate baseline models
    train_evaluate_baseline_models(X_train_scaled, X_val_scaled, X_holdout_scaled,
                                 y_train, y_val, y_holdout)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    X_holdout_tensor = torch.tensor(X_holdout_scaled, dtype=torch.float32)
    y_holdout_tensor = torch.tensor(y_holdout.values, dtype=torch.long)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train teacher model
    input_dim = X_train_scaled.shape[1]
    teacher_model = TeacherModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(teacher_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("\nTraining Teacher Model:")
    train_losses, val_losses = train_model(
        teacher_model, train_loader, val_loader,
        criterion, optimizer, num_epochs=50, patience=3
    )
    
    # Initialize and train student model
    student_model = StudentModel(input_dim)
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("\nTraining Student Model with Knowledge Distillation:")
    train_student_with_distillation(
        student_model, teacher_model, train_loader, val_loader,
        optimizer_student, T=2.0, alpha=0.5, num_epochs=20, patience=3
    )
    
    # Evaluate all models on holdout set
    print("\n=== Final Evaluation on Holdout Set ===")
    
    # Teacher Model Evaluation
    teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = teacher_model(X_holdout_tensor)
        _, teacher_predicted = torch.max(teacher_outputs, 1)
        teacher_accuracy = (teacher_predicted == y_holdout_tensor).float().mean()
        print("\nTeacher Model Metrics:")
        print("Accuracy:", teacher_accuracy.item())
        print("\nClassification Report:")
        print(classification_report(y_holdout_tensor, teacher_predicted))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_holdout_tensor, teacher_predicted))
    
    # Student Model Evaluation
    student_model.eval()
    with torch.no_grad():
        student_outputs = student_model(X_holdout_tensor)
        _, student_predicted = torch.max(student_outputs, 1)
        student_accuracy = (student_predicted == y_holdout_tensor).float().mean()
        print("\nStudent Model Metrics:")
        print("Accuracy:", student_accuracy.item())
        print("\nClassification Report:")
        print(classification_report(y_holdout_tensor, student_predicted))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_holdout_tensor, student_predicted))
    
    # Compare predictions between Teacher and Student
    agreement = (teacher_predicted == student_predicted).float().mean()
    print("\nTeacher-Student Agreement Rate:", agreement.item())
    
    # Analyze mistakes
    disagreement_mask = teacher_predicted != student_predicted
    if disagreement_mask.any():
        print("\nAnalyzing disagreements between Teacher and Student models:")
        print("Number of disagreements:", disagreement_mask.sum().item())
        print("Disagreement indices:", torch.where(disagreement_mask)[0].tolist())
    
    # Model Size Comparison
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    print("\nModel Size Comparison:")
    print(f"Teacher Model Parameters: {count_parameters(teacher_model):,}")
    print(f"Student Model Parameters: {count_parameters(student_model):,}")
    
    # Save results to file
    results = {
        'data_stats': {
            'total_samples': len(data),
            'feature_count': X.shape[1],
            'class_distribution': y.value_counts().to_dict()
        },
        'model_performance': {
            'teacher': {
                'accuracy': teacher_accuracy.item(),
                'parameters': count_parameters(teacher_model)
            },
            'student': {
                'accuracy': student_accuracy.item(),
                'parameters': count_parameters(student_model)
            },
            'agreement_rate': agreement.item()
        }
    }
    
    # Save results to JSON file
    import json
    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nResults have been saved to 'model_evaluation_results.json'")
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Teacher confusion matrix
    teacher_cm = confusion_matrix(y_holdout_tensor, teacher_predicted)
    ax1.imshow(teacher_cm, cmap='Blues')
    ax1.set_title('Teacher Model\nConfusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Student confusion matrix
    student_cm = confusion_matrix(y_holdout_tensor, student_predicted)
    im = ax2.imshow(student_cm, cmap='Blues')
    ax2.set_title('Student Model\nConfusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    
    # Add colorbar
    plt.colorbar(im, ax=(ax1, ax2))
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("\nConfusion matrices have been saved to 'confusion_matrices.png'")

if __name__ == "__main__":
    main() 
