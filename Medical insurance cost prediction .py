"""
Medical Insurance Cost Prediction
AI-Powered Healthcare Cost Estimation using Linear Regression

Author: Your Name
Project: Medical Insurance AI
Description: Predict annual medical insurance costs based on demographic and health factors
"""

# ==========================================
# IMPORT REQUIRED LIBRARIES
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🏥 MEDICAL INSURANCE COST PREDICTION AI 🏥")
print("="*60)
print()

# ==========================================
# STEP 1: DATA COLLECTION & LOADING
# ==========================================

print("📊 Loading Insurance Dataset...")
# Load the insurance dataset
df = pd.read_csv('insurance.csv')
print("✅ Dataset Loaded Successfully!")
print()

# ==========================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

print("="*60)
print("🔍 EXPLORATORY DATA ANALYSIS")
print("="*60)
print()

# Display first 5 rows
print("📋 First 5 Rows of Dataset:")
print(df.head())
print()

# Dataset shape
print(f"📏 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print()

# Dataset info
print("ℹ️ Dataset Information:")
print(df.info())
print()

# Statistical summary
print("📈 Statistical Summary:")
print(df.describe())
print()

# Check for missing values
print("⚠️ Missing Values Check:")
print(df.isnull().sum())
print()

# ==========================================
# STEP 3: DATA VISUALIZATION
# ==========================================

print("="*60)
print("📊 GENERATING VISUALIZATIONS")
print("="*60)
print()

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Age Distribution
print("📈 Creating Age Distribution Plot...")
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=20, color='#00b4d8', edgecolor='black', alpha=0.7)
plt.xlabel('Age', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Age Distribution of Insurance Customers', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Age Distribution saved!")

# 2. Gender Distribution
print("📈 Creating Gender Distribution Plot...")
plt.figure(figsize=(8, 6))
sex_counts = df['sex'].value_counts()
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', 
        colors=['#0077b6', '#00b4d8'], startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Gender Distribution', fontsize=14, fontweight='bold')
plt.savefig('sex_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Gender Distribution saved!")

# 3. BMI Distribution
print("📈 Creating BMI Distribution Plot...")
plt.figure(figsize=(10, 6))
plt.hist(df['bmi'], bins=25, color='#023e8a', edgecolor='black', alpha=0.7)
plt.xlabel('BMI (kg/m²)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('BMI Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('bmi_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ BMI Distribution saved!")

# 4. Children Distribution
print("📈 Creating Children Distribution Plot...")
plt.figure(figsize=(10, 6))
children_counts = df['children'].value_counts().sort_index()
plt.bar(children_counts.index, children_counts.values, color='#0077b6', edgecolor='black', alpha=0.8)
plt.xlabel('Number of Children', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Number of Children', fontsize=14, fontweight='bold')
plt.xticks(children_counts.index)
plt.grid(axis='y', alpha=0.3)
plt.savefig('children_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Children Distribution saved!")

# 5. Smoker Distribution
print("📈 Creating Smoker Distribution Plot...")
plt.figure(figsize=(8, 6))
smoker_counts = df['smoker'].value_counts()
colors = ['#00b4d8', '#E74C3C']  # Blue for no, Red for yes
plt.pie(smoker_counts, labels=smoker_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Smoker Distribution', fontsize=14, fontweight='bold')
plt.savefig('smoker_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Smoker Distribution saved!")

# 6. Charges Distribution
print("📈 Creating Insurance Charges Distribution Plot...")
plt.figure(figsize=(12, 6))
plt.hist(df['charges'], bins=30, color='#00b4d8', edgecolor='black', alpha=0.7)
plt.xlabel('Insurance Charges ($)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Insurance Charges', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('charges_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Charges Distribution saved!")

# 7. Correlation Heatmap
print("📈 Creating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            linewidths=1, linecolor='black', fmt='.2f', 
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Correlation Heatmap saved!")

print()

# ==========================================
# STEP 4: DATA PREPROCESSING
# ==========================================

print("="*60)
print("⚙️ DATA PREPROCESSING")
print("="*60)
print()

# Encoding categorical variables
print("🔄 Encoding Categorical Variables...")

# Encoding Sex: male -> 0, female -> 1
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

# Encoding Smoker: yes -> 0, no -> 1
df['smoker'] = df['smoker'].map({'yes': 0, 'no': 1})

# Encoding Region: southeast -> 0, southwest -> 1, northeast -> 2, northwest -> 3
df['region'] = df['region'].map({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3})

print("✅ Categorical variables encoded successfully!")
print()

# Display encoded data
print("📋 Encoded Data Sample:")
print(df.head())
print()

# ==========================================
# STEP 5: FEATURE-TARGET SEPARATION
# ==========================================

print("="*60)
print("✂️ SEPARATING FEATURES AND TARGET")
print("="*60)
print()

# Separating features (X) and target variable (Y)
X = df.drop(columns='charges', axis=1)
Y = df['charges']

print(f"📊 Features Shape: {X.shape}")
print(f"🎯 Target Shape: {Y.shape}")
print()

# ==========================================
# STEP 6: TRAIN-TEST SPLIT
# ==========================================

print("="*60)
print("📋 SPLITTING DATA INTO TRAINING AND TESTING SETS")
print("="*60)
print()

# Splitting the data: 80% training, 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"📊 Training Set Size: {X_train.shape[0]} samples")
print(f"📊 Testing Set Size: {X_test.shape[0]} samples")
print()

# ==========================================
# STEP 7: MODEL TRAINING
# ==========================================

print("="*60)
print("🤖 TRAINING LINEAR REGRESSION MODEL")
print("="*60)
print()

# Create and train the Linear Regression model
model = LinearRegression()

print("⏳ Training in progress...")
model.fit(X_train, Y_train)
print("✅ Model Training Completed!")
print()

# ==========================================
# STEP 8: MODEL EVALUATION
# ==========================================

print("="*60)
print("📊 MODEL EVALUATION")
print("="*60)
print()

# Predictions on training data
train_predictions = model.predict(X_train)

# Training metrics
train_r2_score = metrics.r2_score(Y_train, train_predictions)
train_mse = metrics.mean_squared_error(Y_train, train_predictions)
train_mae = metrics.mean_absolute_error(Y_train, train_predictions)

print("🎯 TRAINING SET PERFORMANCE:")
print(f"   R² Score: {train_r2_score:.4f} ({train_r2_score*100:.2f}%)")
print(f"   Mean Squared Error: ${train_mse:,.2f}")
print(f"   Mean Absolute Error: ${train_mae:,.2f}")
print()

# Predictions on testing data
test_predictions = model.predict(X_test)

# Testing metrics
test_r2_score = metrics.r2_score(Y_test, test_predictions)
test_mse = metrics.mean_squared_error(Y_test, test_predictions)
test_mae = metrics.mean_absolute_error(Y_test, test_predictions)

print("🎯 TESTING SET PERFORMANCE:")
print(f"   R² Score: {test_r2_score:.4f} ({test_r2_score*100:.2f}%)")
print(f"   Mean Squared Error: ${test_mse:,.2f}")
print(f"   Mean Absolute Error: ${test_mae:,.2f}")
print()

# 8. Prediction vs Actual Plot
print("📈 Creating Prediction vs Actual Plot...")
plt.figure(figsize=(12, 6))
plt.scatter(Y_test, test_predictions, alpha=0.6, color='#00b4d8', edgecolors='#023e8a')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Charges ($)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Charges ($)', fontsize=12, fontweight='bold')
plt.title('Actual vs Predicted Insurance Charges', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Prediction vs Actual Plot saved!")
print()

# ==========================================
# STEP 9: MAKING PREDICTIONS
# ==========================================

print("="*60)
print("💉 MAKING PREDICTIONS ON NEW DATA")
print("="*60)
print()

# Example: Predict insurance cost for a new customer
# Input: [age, sex, bmi, children, smoker, region]
# Example: 31 year old female, BMI 25.74, no children, non-smoker, southeast region

input_data = (31, 1, 25.74, 0, 1, 0)

# Convert to numpy array
input_data_array = np.asarray(input_data)

# Reshape for prediction
input_data_reshaped = input_data_array.reshape(1, -1)

# Make prediction
prediction = model.predict(input_data_reshaped)

print("📋 Input Data:")
print(f"   Age: {input_data[0]} years")
print(f"   Sex: {'Female' if input_data[1] == 1 else 'Male'}")
print(f"   BMI: {input_data[2]} kg/m²")
print(f"   Children: {input_data[3]}")
print(f"   Smoker: {'No' if input_data[4] == 1 else 'Yes'}")
print(f"   Region: {['Southeast', 'Southwest', 'Northeast', 'Northwest'][input_data[5]]}")
print()
print(f"💰 Predicted Annual Insurance Cost: ${prediction[0]:,.2f}")
print()

# ==========================================
# STEP 10: SAVE PREDICTIONS TO CSV
# ==========================================

print("="*60)
print("💾 SAVING PREDICTIONS")
print("="*60)
print()

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'Actual_Charges': Y_test.values,
    'Predicted_Charges': test_predictions,
    'Age': X_test['age'].values,
    'Sex': X_test['sex'].values,
    'BMI': X_test['bmi'].values,
    'Children': X_test['children'].values,
    'Smoker': X_test['smoker'].values,
    'Region': X_test['region'].values
})

# Save to CSV
predictions_df.to_csv('insurance_predictions.csv', index=False)
print("✅ Predictions saved to 'insurance_predictions.csv'")
print()

# ==========================================
# STEP 11: SAVE THE MODEL
# ==========================================

print("="*60)
print("💾 SAVING TRAINED MODEL")
print("="*60)
print()

import pickle

# Save the trained model
with open('insurance_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved as 'insurance_prediction_model.pkl'")
print()

# ==========================================
# KEY INSIGHTS
# ==========================================

print("="*60)
print("🔬 KEY INSIGHTS FROM DATA")
print("="*60)
print()

# Smoking impact
smoker_yes = df[df['smoker'] == 0]['charges'].mean()
smoker_no = df[df['smoker'] == 1]['charges'].mean()
smoking_difference = smoker_yes - smoker_no

print(f"🚬 SMOKING IMPACT:")
print(f"   Average cost (Smokers): ${smoker_yes:,.2f}/year")
print(f"   Average cost (Non-smokers): ${smoker_no:,.2f}/year")
print(f"   Difference: ${abs(smoking_difference):,.2f} ({abs(smoking_difference)/smoker_no*100:.1f}% more for smokers!)")
print()

# Age correlation
age_corr = df[['age', 'charges']].corr().iloc[0, 1]
print(f"📊 AGE CORRELATION: {age_corr:.3f} (Strong positive correlation)")
print()

# Gender impact
male_avg = df[df['sex'] == 0]['charges'].mean()
female_avg = df[df['sex'] == 1]['charges'].mean()
print(f"👥 GENDER IMPACT:")
print(f"   Average cost (Male): ${male_avg:,.2f}/year")
print(f"   Average cost (Female): ${female_avg:,.2f}/year")
print()

# Regional differences
print(f"🗺️ REGIONAL VARIATIONS:")
regions = ['Southeast', 'Southwest', 'Northeast', 'Northwest']
for i, region in enumerate(regions):
    avg_cost = df[df['region'] == i]['charges'].mean()
    print(f"   {region}: ${avg_cost:,.2f}/year")
print()

# ==========================================
# COMPLETION MESSAGE
# ==========================================

print("="*60)
print("✅ MEDICAL INSURANCE PREDICTION COMPLETE!")
print("="*60)
print()
print("📊 Generated Files:")
print("   ✅ age_distribution.png")
print("   ✅ sex_distribution.png")
print("   ✅ bmi_distribution.png")
print("   ✅ children_distribution.png")
print("   ✅ smoker_distribution.png")
print("   ✅ charges_distribution.png")
print("   ✅ correlation_heatmap.png")
print("   ✅ prediction_vs_actual.png")
print("   ✅ insurance_predictions.csv")
print("   ✅ insurance_prediction_model.pkl")
print()
print("🏥 Model Accuracy: {:.2f}%".format(test_r2_score * 100))
print("💊 Ready for deployment!")
print()
print("="*60)
