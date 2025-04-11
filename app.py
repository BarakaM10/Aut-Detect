from flask import Flask, request, render_template, send_file
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import io

app = Flask(__name__)

# Model definition
class StudentModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout_rate=0.2):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.adaptation_layer = nn.Linear(hidden_dim, num_classes)
        self.is_adapted = False

    def forward(self, x):
        features = self.dropout(self.bn1(torch.relu(self.fc1(x))))
        return self.adaptation_layer(features) if self.is_adapted else self.fc2(features)

# Load model and scaler
with open("model/model_config.json") as f:
    config = json.load(f)

model = StudentModel(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    dropout_rate=config['dropout_rate']
)
model.load_state_dict(torch.load("model/student_model.pth", map_location=torch.device('cpu')))
model.eval()

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_names = config["feature_names"]
results_df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global results_df
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    df = pd.read_csv(file)

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        return f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}", 400

    X = df[feature_names].copy()
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    results = []
    with torch.no_grad():
        for i in range(len(X_tensor)):
            output = model(X_tensor[i:i+1])
            probs = torch.softmax(output, dim=1).numpy()[0]
            prediction = int(torch.argmax(output, dim=1).item())
            result = {
                "image_id": df.iloc[i].get("image_id", f"Row_{i+1}"),
                "actual_label": df.iloc[i].get("autism", "Unknown"),
                "prediction": "Autism" if prediction == 1 else "Non-Autism",
                "autism_prob": round(probs[1], 4),
                "non_autism_prob": round(probs[0], 4),
                "confidence": round(max(probs), 4)
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    avg_conf = round(results_df["confidence"].mean(), 4)
    low_conf = len(results_df[results_df["confidence"] < 0.6])

    return render_template("index.html", results=results, avg_conf=avg_conf, low_conf=low_conf)

@app.route('/download')
def download():
    global results_df
    if results_df is None:
        return "No results to download", 400

    output = io.StringIO()
    results_df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='predictions.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
