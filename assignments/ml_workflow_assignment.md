# ML Workflow Assignment

## Problem Statement
You are given a dataset of customer orders and need to predict whether a customer will make a repeat purchase within 30 days.

---

##  Task 1

### Label Column
**`repeat_purchase_flag`**

**Justification:**  
This column represents the outcome we want to predict — whether a customer makes a repeat purchase within 30 days — making it the target variable for the model.

---

### Data Leakage Column
**`discount_used_on_repeat_order`**

**Justification:**  
This column contains information about the repeat purchase itself, which would not be available at the time of prediction and would lead to data leakage if used as a feature.

---

##  Task 2

### Step 1: Exploratory Data Analysis (EDA)

**Explanation:**  
EDA helps in understanding the structure of the dataset, identifying missing values, detecting outliers, and analyzing relationships between variables. This step ensures that the data is suitable for modeling and helps guide feature selection.

---

### Step 2: Data Preprocessing (Cleaning & Feature Engineering)

**Explanation:**  
Data preprocessing prepares the dataset for modeling by handling missing values, removing irrelevant or leakage features, and selecting meaningful input variables. Proper preprocessing improves model performance and prevents misleading results.

---

##  Additional Notes

- The leakage column (`discount_used_on_repeat_order`) was removed before training the model.
- The dataset was split into training and testing sets using an 80-20 split.
- A Gradient Boosting Classifier was used for prediction.

---

##  Conclusion

A proper machine learning workflow involves understanding the data, cleaning and preparing it, and only then applying models. Skipping these steps can lead to inaccurate and unreliable results.