import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from transformers import DistilBertTokenizer, DistilBertModel
import re

# ---------------------------- Load all Models ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = joblib.load("scaler_d.pkl")
scaler_bert = joblib.load("scaler_bert_d.pkl")
label_encoder = joblib.load("label_encoder_d.pkl")
severity_model = joblib.load("xgb_model_fd.pkl")

# ---------------------------- Severity Mapping ----------------------------
def map_severity(severity_num):
    severity_mapping = {
        0: "Low",
        1: "Medium",
        2: "High",
        3: "Critical",
        4: "Urgent",
        5: "Warning",
        6: "Info",
        7: "Notice",
        8: "Emergency",
        9: "Disaster",
        10: "high",
        11: "high",
        12: "high",
        
    }
    return severity_mapping.get(severity_num, "High")  # Default to High if unknown

# ---------------------------- LSTM Model Definition ----------------------------
class BalancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BalancedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=256, num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc1 = nn.Linear(256 * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.out(out)

# Load LSTM Model
input_size = 781
num_classes = len(label_encoder.classes_)
lstm_model = BalancedLSTM(input_size=input_size, hidden_size=256, num_classes=num_classes).to(device)
lstm_model.load_state_dict(torch.load("lstm_model_d.pth", map_location=device))
lstm_model.eval()

# Load BERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device).eval()

# ---------------------------- Feature Extraction ----------------------------
def extract_features(log_messages, cpu_usage, mem_usage):
    inputs = tokenizer(log_messages, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    bert_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    bert_scaled = scaler_bert.transform(bert_embeddings)

    df = pd.DataFrame({'Log_Message': log_messages})
    df['CPU_Usage (%)'] = cpu_usage
    df['Memory_Usage (%)'] = mem_usage
    df['hour'] = 12
    df['minute'] = 30
    df['day_of_week'] = 2
    df['is_weekend'] = 0
    df['log_length'] = df['Log_Message'].astype(str).apply(len)
    df['word_count'] = df['Log_Message'].astype(str).apply(lambda x: len(x.split()))
    df['char_count'] = df['Log_Message'].astype(str).apply(len)
    df['upper_case_ratio'] = df['Log_Message'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    df['error_freq'] = df['Log_Message'].str.count(r'error|fail|critical|exception', flags=re.IGNORECASE)
    df['error_keywords'] = df['Log_Message'].astype(str).str.contains(r"error|fail|exception", case=False).astype(int)

    numeric_scaled = scaler.transform(df[['CPU_Usage (%)', 'Memory_Usage (%)']])
    system_load = ((numeric_scaled[:, 0] + numeric_scaled[:, 1]) / 2).reshape(-1, 1)
    other_numeric = df[['hour', 'minute', 'day_of_week', 'is_weekend', 'log_length', 'word_count', 'char_count', 'upper_case_ratio', 'error_freq', 'error_keywords']].values
    numeric_final = np.hstack([numeric_scaled, system_load, other_numeric])

    final_features = np.hstack([bert_scaled, numeric_final])
    return final_features

# ---------------------------- Action & Recommendation Mapping ----------------------------
expanded_action_mapping = {
    "Investigate logs": "Reset credentials, check firewall settings, and review access logs",
    "Increase memory limits": "Scale application memory or optimize memory-intensive tasks",
    "Restart application": "Check logs for crash reason and restart the service/container",
    "Check database credentials": "Validate DB username/password and service connectivity",
    "No Action Required": "No immediate intervention needed",
    "Optimize background tasks": "Analyze and tune background jobs or schedulers",
    "Check network connectivity": "Ping upstream devices, test DNS resolution, verify firewall rules",
    "Apply security patch": "Patch vulnerabilities and restart affected services",
    "Expand storage capacity": "Increase disk allocation or mount additional storage volumes",
    "Clear temporary files": "Delete temporary files and clear application caches",
    "Restart services or scale resources": "Restart affected services or scale compute resources as needed",
    "Scale compute resources": "Provision additional CPU or memory to manage load spikes",
    "Rollback last deployment": "Revert to a stable application version and monitor system stability",
    "Inspect firewall rules": "Review and modify firewall settings to allow required traffic",
    "Restart application server": "Restart the web/application server and monitor its health",
    "Optimize database queries": "Identify and improve slow database queries",
    "Restart network services": "Restart network daemons and check connectivity",
    "Archive old logs": "Move old log files to long-term storage or delete them to free space",
    "Review heap memory allocations": "Analyze heap usage and optimize memory allocation",
    "Perform hardware diagnostics": "Run system-level diagnostics to identify potential hardware issues",
    "Block IP address": "Block suspicious IP addresses at the firewall or IDS/IPS level",
    "Investigate runaway process": "Identify and terminate processes consuming excessive resources",
    "Restart database service": "Restart the database service and verify data integrity",
}

def generate_default_description(action_label):
    label = action_label.lower()
    if "restart" in label:
        return f"Restart the component related to '{action_label}'"
    elif "check" in label:
        return f"Investigate system components and logs for '{action_label}'"
    elif "update" in label:
        return f"Ensure latest version of software or config for '{action_label}'"
    elif "clean" in label or "flush" in label:
        return f"Perform cleanup operation related to '{action_label}'"
    elif "investigate" in label or "review" in label:
        return f"Deep dive into '{action_label}' using monitoring and logging tools"
    else:
        return f"Perform appropriate diagnostic and resolution steps for '{action_label}'"

def get_recommendation(pred_label):
    return expanded_action_mapping.get(pred_label, generate_default_description(pred_label))

# ---------------------------- Streamlit App ----------------------------
st.set_page_config(page_title="Log Action Predictor", layout="wide")
st.title("Log Suggested Action Predictor")

log_input = st.text_area("Paste your log messages (one per line):", height=200)
cpu_input = st.number_input("CPU Usage (%)", 0.0, 100.0, 50.0)
mem_input = st.number_input("Memory Usage (%)", 0.0, 100.0, 50.0)

if st.button("Predict"):
    log_lines = [line.strip() for line in log_input.strip().split('\n') if line.strip()]

    if log_lines:
        features = extract_features(log_lines, [cpu_input] * len(log_lines), [mem_input] * len(log_lines))

        # LSTM Action Prediction
        features_seq = features.reshape(features.shape[0], 1, features.shape[1])
        torch_X = torch.tensor(features_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = lstm_model(torch_X)
            _, preds = torch.max(outputs, 1)
        predicted_actions = label_encoder.inverse_transform(preds.cpu().numpy())

        # Recommended Action Mapping
        recommended_actions = [get_recommendation(act) for act in predicted_actions]

        # Severity Prediction
        severity_preds = severity_model.predict(features)

        # 🔹 Map severity numbers to labels
        predicted_severity = [map_severity(sev) for sev in severity_preds]

        # Display Results
        output_df = pd.DataFrame({
            "Log Message": log_lines,
            "Predicted Action": predicted_actions,
            "Recommended Action": recommended_actions,
            "Predicted Severity": predicted_severity,
        })

        st.write("### Prediction Results:")
        st.dataframe(output_df, use_container_width=True)
        st.download_button("Download Results as CSV", output_df.to_csv(index=False), file_name="log_predictions.csv")
    else:
        st.warning("Please paste some log messages.")
