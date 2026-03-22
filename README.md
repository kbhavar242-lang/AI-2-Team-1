📌 Project Title
Insurance Cost Prediction – Estimating Medical Charges Using Linear Regression
👥 Team Members
Team Leader: Krishna Bhavar (Integration & Documentation)

Member 1: Agaram Chandra Has Reddy (Data Loading & Exploration)

Member 2: Aryan Choudhary (Preprocessing & Cleaning)

Member 3: B Yashwanth Reddy (Train-Test Split & Linear Model)

Member 4: Banoth Mahesh (Evaluation & Metrics)

Member 5: Bhadane Sayali (Model Improvements)

Member 6: Amal K (Merging & Final Integration)

🎯 Problem Statement
The goal of this project is to design a basic machine learning pipeline capable of predicting individual medical insurance costs based on personal and lifestyle attributes.

This project demonstrates the complete workflow of a supervised regression problem, including:

Data loading and exploration

Data preprocessing and cleaning

Model training using linear regression

Performance evaluation

Basic improvements to enhance results

Additionally, the project emphasizes collaborative development using Git and GitHub, following a structured branch-based workflow along with pull requests for code integration.

📊 Dataset Description
The project uses the Medical Cost Personal Dataset (Insurance Dataset), which contains information about individuals along with their annual medical insurance expenses.

Each record corresponds to one individual, where the target variable charges represents the insurance cost.

🔑 Features (Columns)
age: Age of the individual (numeric)

sex: Gender of the individual (categorical: male/female)

bmi: Body Mass Index – an indicator of body fat (numeric)

children: Number of dependents (integer)

smoker: Smoking status (categorical: yes/no)

region: Residential region (categorical: northeast, northwest, southeast, southwest)

charges: Medical insurance cost (continuous – target variable)

⚙️ Data Preprocessing Steps
🔹 1. Data Loading & Initial Exploration
Loaded dataset using pandas from insurance_data_linear.csv

Examined dataset structure, data types, and summary statistics

Checked for missing values

🔹 2. Handling Missing Values
Verified missing entries column-wise

Removed rows with missing values (if present) using dropna() for simplicity

🔹 3. Feature Selection
Selected input features:
age, bmi, children, sex, smoker, region

Selected target variable:
charges

🔹 4. Encoding Categorical Variables
Applied One-Hot Encoding using OneHotEncoder within a ColumnTransformer

Used drop="first" to prevent multicollinearity caused by dummy variables

🔹 5. Feature Scaling (Improved Model)
Applied StandardScaler to numerical features after adding polynomial features

Scaled features include:
age, bmi, children

🔹 6. Train-Test Split
Split dataset into training and testing sets using train_test_split

Configuration used:
80% training / 20% testing

🤖 Model Used & Training Details
We implemented Linear Regression, a supervised learning algorithm used to model the relationship between input features and the target variable (charges).

The model learns a linear relationship that helps estimate insurance costs based on the given attributes.

🚀 Summary
This project provides a complete pipeline for predicting medical insurance costs while also demonstrating:

Practical machine learning workflow

Data preprocessing techniques

Model evaluation strategies

Collaborative development using Git branches
