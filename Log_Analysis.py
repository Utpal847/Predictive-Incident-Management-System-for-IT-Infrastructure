import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import torch
import re

# Load dataset
file_path = "/content/drive/MyDrive/log_data_reduced.csv"
df = pd.read_csv(file_path)


# Preprocessing
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp'])
df['Incident_Label'] = df['Incident_Label'].fillna('None')
df['Severity'] = df['Severity'].fillna('Low')
df['Suggested_Action'] = df['Suggested_Action'].replace([
    'None', 'none', 'NONE', None, np.nan
], 'No Action Required')
df['hour'] = df['Timestamp'].dt.hour
df['minute'] = df['Timestamp'].dt.minute
df['day_of_week'] = df['Timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['log_length'] = df['Log_Message'].astype(str).apply(len)
df['word_count'] = df['Log_Message'].astype(str).apply(lambda x: len(x.split()))
df['char_count'] = df['Log_Message'].astype(str).apply(len)
df['upper_case_ratio'] = df['Log_Message'].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
df['error_freq'] = df['Log_Message'].str.count(r'error|fail|critical|exception', flags=re.IGNORECASE)
df['error_keywords'] = df['Log_Message'].astype(str).str.contains("error|fail|exception", case=False).astype(int)

# Train/test split
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['Suggested_Action'], random_state=42
)

# Encode labels
label_encoder = LabelEncoder()
train_df['Suggested_Action_Encoded'] = label_encoder.fit_transform(train_df['Suggested_Action'])
test_df['Suggested_Action_Encoded'] = label_encoder.transform(test_df['Suggested_Action'])

# Normalize numerical features
scaler = StandardScaler()
numeric_cols = ['CPU_Usage (%)', 'Memory_Usage (%)']
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
train_df['system_load'] = (train_df['CPU_Usage (%)'] + train_df['Memory_Usage (%)']) / 2
test_df['system_load'] = (test_df['CPU_Usage (%)'] + test_df['Memory_Usage (%)']) / 2

# Load BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.to(device).eval()

# Feature extraction using [CLS] token
def extract_features(df_part):
    embeddings, numeric_feats, labels = [], [], []
    BATCH_SIZE = 16
    for i in tqdm(range(0, len(df_part), BATCH_SIZE)):
        batch = df_part.iloc[i:i+BATCH_SIZE]
        texts = batch['Log_Message'].astype(str).tolist()
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        numeric = batch[[
            'CPU_Usage (%)', 'Memory_Usage (%)', 'hour', 'minute', 'system_load',
            'day_of_week', 'is_weekend', 'log_length', 'word_count', 'char_count',
            'upper_case_ratio', 'error_freq', 'error_keywords'
        ]].values
        embeddings.append(pooled)
        numeric_feats.append(numeric)
        labels.append(batch['Suggested_Action_Encoded'].values)
    return (
        np.vstack(embeddings),
        np.vstack(numeric_feats),
        np.concatenate(labels)
    )

X_train_bert, X_train_numeric, y_train = extract_features(train_df)
X_test_bert, X_test_numeric, y_test = extract_features(test_df)

# Combine features
scaler_bert = StandardScaler()
X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert)
X_test_bert_scaled = scaler_bert.transform(X_test_bert)
X_train_final = np.hstack((X_train_bert_scaled, X_train_numeric))
X_test_final = np.hstack((X_test_bert_scaled, X_test_numeric))

# -------------------- Balance Data --------------------
X_train_bal = X_train_final
y_train_bal = y_train

#Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced_subsample',
    random_state=42
)
rf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf.predict(X_test_final)

print("\n📊 Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))


from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#XGBoost

xgb = XGBClassifier(
    n_estimators=100,       # reduce from 200
    max_depth=6,            # shallower trees = faster
    learning_rate=0.1,      # slightly higher LR
    use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),

    random_state=42
)

xgb.fit(
    X_train_bal,
    y_train_bal,
    eval_set=[(X_test_final, y_test)],
    verbose=True
)

y_pred_xgb = xgb.predict(X_test_final)

print("\n📊 XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))


#LSTM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ----------------------------
# Step 1: Reshape Data and Split into Train/Validation
# ----------------------------
X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(
    X_train_bal, y_train_bal, test_size=0.2, stratify=y_train_bal, random_state=42
)

X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], 1, X_train_seq.shape[1]))
X_val_seq = X_val_seq.reshape((X_val_seq.shape[0], 1, X_val_seq.shape[1]))
X_test_lstm = X_test_final.reshape((X_test_final.shape[0], 1, X_test_final.shape[1]))

torch_X_train = torch.tensor(X_train_seq, dtype=torch.float32)
torch_y_train = torch.tensor(y_train_seq, dtype=torch.long)
torch_X_val = torch.tensor(X_val_seq, dtype=torch.float32)
torch_y_val = torch.tensor(y_val_seq, dtype=torch.long)
torch_X_test = torch.tensor(X_test_lstm, dtype=torch.float32)
torch_y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(torch_X_train, torch_y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(torch_X_val, torch_y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(TensorDataset(torch_X_test, torch_y_test), batch_size=64, shuffle=False)

# ----------------------------
# Step 2: Define LSTM Model
# ----------------------------
class BalancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BalancedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.out(out)

# ----------------------------
# Step 3: Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_train_bal.shape[1]
hidden_size = 256
num_classes = len(np.unique(y_train_bal))

model = BalancedLSTM(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ----------------------------
# Step 4: Training with Early Stopping
# ----------------------------
best_val_loss = float('inf')
patience = 7
patience_counter = 0
best_model_state = None
epochs = 200

print("\n🚀 Training with Early Stopping...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_X, val_y in val_loader:
            val_X, val_y = val_X.to(device), val_y.to(device)
            val_outputs = model(val_X)
            val_loss += criterion(val_outputs, val_y).item()

    val_loss /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹ Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)

# ----------------------------
# Step 5: Evaluation
# ----------------------------
model.eval()
y_pred_lstm = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        y_pred_lstm.extend(predicted.cpu().numpy())

# ----------------------------
# Step 6: Metrics & Visualization
# ----------------------------
print("\n📊 Classification Report (Test Set):")
print(classification_report(y_test, y_pred_lstm, target_names=label_encoder.classes_))
print("✅ Accuracy:", accuracy_score(y_test, y_pred_lstm))



import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ✅ 1. Decode LSTM predictions using LabelEncoder
test_df["Predicted_Suggested_Action"] = label_encoder.inverse_transform(y_pred_lstm)

# ✅ 2. Define initial expanded action explanations
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
    "Clean up disk": "Delete old logs, compress backups, or extend volume space",
    "Reboot system": "Ensure safe shutdown, check system logs, and restart hardware cleanly",
    "Update software package": "Run update commands and review changelog or dependency versions",
    "Check load balancer status": "Inspect load balancer health and traffic distribution logic",
    "Restart JVM": "Investigate heap usage and restart Java Virtual Machine service",
    "Review system configuration": "Validate config files for deprecated or misconfigured keys",
    "Restart container": "Stop and start the affected Docker/Kubernetes container",
    "Check certificate validity": "Renew expired certificates and reload services if needed",
    "Check disk IOPS": "Diagnose slow I/O operations using system metrics and SMART tools",
    "Scale horizontally": "Add more application instances or VMs to handle load",
    "Enable alerting": "Configure monitoring tools to trigger alerts on similar incidents",
    "Update firewall rules": "Adjust inbound/outbound firewall rules to allow expected traffic",
    "Check system time": "Verify NTP sync and timezone for consistent logging and SSL validity",
}



# ✅ 3. Auto-generate fallback description for new predicted actions
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

# ✅ 4. Extend mapping dynamically
unique_preds = test_df["Predicted_Suggested_Action"].unique()
for pred in unique_preds:
    if pred not in expanded_action_mapping:
        expanded_action_mapping[pred] = generate_default_description(pred)

# ✅ 5. Build backup (fallback) mapping from training data
fallback_mapping = (
    train_df.groupby("Suggested_Action")["Suggested_Action"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "⚠️ Needs Review")
    .to_dict()
)

# ✅ 6. Combine mappings into final function
def get_recommendation(pred_label):
    if pred_label in expanded_action_mapping:
        return expanded_action_mapping[pred_label]
    elif pred_label in fallback_mapping:
        return fallback_mapping[pred_label]
    else:
        return "⚠️ Needs Review - Not Mapped"

# ✅ 7. Apply the logic
test_df["Recommended_Action"] = test_df["Predicted_Suggested_Action"].apply(get_recommendation)

# ✅ 8. Train severity classifier using RF and XGB
X_severity_train = X_train_final
X_severity_test = X_test_final

y_severity_train = train_df['Severity']
y_severity_test = test_df['Severity']

# Encode severity labels
severity_encoder = LabelEncoder()
y_train_s = severity_encoder.fit_transform(y_severity_train)
y_test_s = severity_encoder.transform(y_severity_test)

# Random Forest
rf_sev = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf_sev.fit(X_severity_train, y_train_s)
y_pred_rf_s = rf_sev.predict(X_severity_test)

# XGBoost
xgb_sev = XGBClassifier(n_estimators=100, max_depth=6, eval_metric='mlogloss')
xgb_sev.fit(X_severity_train, y_train_s)
y_pred_xgb_s = xgb_sev.predict(X_severity_test)

# ✅ 9. Predict severity for test_df using XGBoost
test_df["Predicted_Severity"] = severity_encoder.inverse_transform(y_pred_xgb_s)

# Display full column width in console
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# Print clean table with selected columns
display_cols = ["Log_Message", "Predicted_Suggested_Action", "Recommended_Action", "Predicted_Severity"]


from IPython.display import display, HTML

# Subset of final DataFrame (10 rows)
preview_df = test_df[display_cols].head(20)

# Rename columns with styled HTML headers (font-size: 16px or larger)
styled_cols = {
    "Log_Message": "<span style='font-size:16px; font-weight:bold;'>Log Message</span>",
    "Predicted_Suggested_Action": "<span style='font-size:16px; font-weight:bold;'>Predicted Action</span>",
    "Recommended_Action": "<span style='font-size:16px; font-weight:bold;'>Recommended Action</span>",
    "Predicted_Severity": "<span style='font-size:16px; font-weight:bold;'>Predicted Severity</span>",
}

# Replace column names
preview_df.columns = [styled_cols[col] for col in preview_df.columns]

# Display styled table
display(HTML(preview_df.to_html(escape=False, index=False)))


# ✅ 10. Save to CSV
test_df[["Log_Message", "Predicted_Suggested_Action", "Recommended_Action", "Predicted_Severity"]].to_csv("final_lstm_action_mapping_with_severity.csv", index=False)



