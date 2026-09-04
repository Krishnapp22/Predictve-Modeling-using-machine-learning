import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    roc_auc_score
)

# =====================================================================
# STEP 1: PROJECT SETUP & DATA PREPARATION
# =====================================================================

# 1. Generate a mock customer dataset
np.random.seed(42)
num_samples = 1000

X = pd.DataFrame({
    'Age': np.random.randint(18, 70, size=num_samples),
    'Tenure': np.random.randint(1, 72, size=num_samples),
    'MonthlyCharges': np.random.uniform(20.0, 120.0, size=num_samples),
    'UsageFrequency': np.random.randint(0, 30, size=num_samples)
})

# Create a baseline target variable (30% churn rate)
y = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

# 2. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Scale features (Required for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================================
# STEP 2: TRAINING THE PREDICTIVE MODELS
# =====================================================================

# 1. Initialize models
log_model = LogisticRegression(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Train (fit) models
log_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)  # Tree-based models don't require scaling

# 3. Generate predictions
y_pred_log = log_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)

# 4. Print baseline accuracy
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.2%}")
print(f"Random Forest Accuracy:       {accuracy_score(y_test, y_pred_rf):.2%}\n")


# =====================================================================
# STEP 3: VISUALIZING PERFORMANCE & ACCURACY
# =====================================================================

# Setup a clean figure layout for two side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Confusion Matrix for the Random Forest model
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Churned'])
disp.plot(cmap=plt.cm.Blues, ax=axes[0])
axes[0].set_title("Random Forest Confusion Matrix")

# Plot 2: ROC Curve for Random Forest
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
auc_score = roc_auc_score(y_test, y_prob_rf)

axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
axes[1].legend(loc="lower right")
axes[1].grid(True, linestyle=':', alpha=0.6)

# Display plots
plt.tight_layout()
plt.show()
