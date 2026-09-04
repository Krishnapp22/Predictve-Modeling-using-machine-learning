# Customer Churn Prediction using Machine Learning

A machine learning project that predicts whether a customer is likely to **churn (leave a service)** using customer demographic and usage-related information.

This project compares the performance of **Logistic Regression** and **Random Forest Classifier** and evaluates the Random Forest model using a confusion matrix and ROC-AUC curve.

---

## 📌 Project Overview

Customer churn prediction is an important machine learning application for businesses that want to identify customers who may discontinue their services.

In this project, a **mock customer dataset containing 1,000 records** is generated. The dataset includes:

* Age
* Tenure
* Monthly Charges
* Usage Frequency
* Churn status

Two classification algorithms are trained:

1. **Logistic Regression**
2. **Random Forest Classifier**

The models are evaluated using accuracy, confusion matrix, and ROC-AUC.

---

## 🎯 Objectives

* Generate a synthetic customer dataset.
* Split the dataset into training and testing sets.
* Scale features for Logistic Regression.
* Train Logistic Regression and Random Forest models.
* Compare model accuracy.
* Visualize Random Forest classification performance.
* Calculate and visualize the ROC curve and AUC score.

---

## 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── customer_churn.py
├── README.md
└── requirements.txt
```

> Rename your Python file to `customer_churn.py` or update the filename above to match your actual file.

---

## 📊 Dataset

The project uses a **synthetically generated dataset** rather than a real-world customer dataset.

The following features are generated:

| Feature          | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `Age`            | Customer age between 18 and 69                          |
| `Tenure`         | Number of months the customer has been with the service |
| `MonthlyCharges` | Customer's monthly service charges                      |
| `UsageFrequency` | Number of times the customer uses the service           |
| `Churn`          | Target variable: `0 = Stayed`, `1 = Churned`            |

The dataset contains:

* **1,000 samples**
* **4 input features**
* Approximately **30% churn rate**

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as a baseline classification model.

Because Logistic Regression is sensitive to feature scales, the input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.

Scaling is not required for tree-based models, so the original features are used:

```python
rf_model.fit(X_train, y_train)
```

---

## 🔄 Machine Learning Workflow

```text
Generate Dataset
       ↓
Data Preparation
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Train Models
       ↓
Generate Predictions
       ↓
Evaluate Accuracy
       ↓
Confusion Matrix
       ↓
ROC Curve & AUC
```

---

## 📈 Model Evaluation

The project calculates the accuracy of both models:

```text
Logistic Regression Accuracy
Random Forest Accuracy
```

The Random Forest model is additionally evaluated using:

### Confusion Matrix

The confusion matrix shows:

* **True Negatives** — Customers correctly predicted to stay
* **True Positives** — Customers correctly predicted to churn
* **False Positives** — Customers predicted to churn but actually stayed
* **False Negatives** — Customers predicted to stay but actually churned

### ROC Curve

The ROC curve evaluates how well the model distinguishes between customers who stay and customers who churn.

The **AUC (Area Under the Curve)** summarizes the model's classification performance.

* AUC = 1.0 → Perfect classification
* AUC = 0.5 → Random guessing
* Higher AUC → Better classification performance

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
```

### 2. Navigate to the project directory

```bash
cd customer-churn-prediction
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

Or, if you create a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Run the Python script

```bash
python customer_churn.py
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
numpy
pandas
matplotlib
scikit-learn
```

---

## 📷 Output

When the program runs, it prints the accuracy of both machine learning models:

```text
Logistic Regression Accuracy: XX.XX%
Random Forest Accuracy:       XX.XX%
```

It also generates a figure containing:

1. **Random Forest Confusion Matrix**
2. **Random Forest ROC Curve with AUC score**

---

## ⚠️ Important Note About the Dataset

The target variable in this project is generated randomly:

```python
y = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
```

Therefore, the customer features are **not actually related to churn**.

This means the models are essentially learning from random noise, and the resulting accuracy/AUC should **not be interpreted as a meaningful real-world churn prediction result**.

For a production-quality project, a real customer churn dataset should be used where customer characteristics have a meaningful relationship with churn.

---

## 🚀 Future Improvements

This project can be extended by:

* Using a real-world customer churn dataset.
* Adding more customer features.
* Performing exploratory data analysis (EDA).
* Handling missing values.
* Handling categorical variables.
* Performing feature engineering.
* Using cross-validation.
* Hyperparameter tuning.
* Comparing additional algorithms such as:

  * XGBoost
  * Support Vector Machine
  * K-Nearest Neighbors
  * Gradient Boosting
* Adding precision, recall, and F1-score.
* Addressing class imbalance.
* Deploying the model using Flask or Streamlit.

---

## 🧠 Key Concepts Demonstrated

This project demonstrates practical knowledge of:

* Supervised Machine Learning
* Binary Classification
* Train/Test Splitting
* Feature Scaling
* Logistic Regression
* Random Forest
* Model Prediction
* Accuracy Evaluation
* Confusion Matrix
* ROC Curve
* ROC-AUC
* Data Visualization

---

## 👨‍💻 Author

**Krishnapranav Prasad**

---

## ⭐ If You Found This Project Useful

Feel free to ⭐ star the repository and use the project as a starting point for experimenting with customer churn prediction and classification algorithms.
