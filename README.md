 🏢 HR Attrition Analysis & Prediction System: 

This project is a machine learning-based web application that analyzes employee data and predicts the likelihood of employee attrition. It is designed to help understand key workforce patterns and demonstrate an end-to-end ML pipeline from data processing to deployment.

 🎯 Objective: 

The goal of this project is to explore how machine learning can be used to identify employees who may be at risk of leaving an organization and understand the factors influencing attrition behavior.

 ⚙️ Project Workflow: 

The project follows a complete ML pipeline:

Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Training → Evaluation → Deployment

📊 Dataset Overview: 

The dataset contains HR-related attributes such as:

- Age
- Monthly Income
- Overtime
- Business Travel
- Job Role
- Years at Company
- Work-Life Balance
- Job Satisfaction

 🧠 Machine Learning Approach: 

- Model Used: Logistic Regression
- Handling Imbalance: Class Weight Balancing
- Feature Encoding: One-Hot Encoding
- Scaling: StandardScaler

The model outputs a probability score indicating the likelihood of attrition.

 📈 Evaluation Metrics: 

- Accuracy: ~75%
- ROC-AUC Score: ~0.80
- Confusion Matrix used for performance analysis

 🌐 Web Application: 

The model is deployed using Streamlit, providing an interactive interface where users can input employee details and get real-time predictions.

🔗 Live Demo:  
https://hr-attrition-analysis-ebypupeuruvpyw88qmbjjm.streamlit.app/

💡 Key Insights: 

The model shows that employee attrition is strongly influenced by:

- Work pressure (overtime)
- Frequent business travel
- Job satisfaction level
- Work-life balance
- Tenure in the company

🛠 Tech Stack: 

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

🚀 Project Structure: 
app_2.py # Streamlit application
model.pkl # Trained ML model
scaler.pkl # Feature scaler
features.pkl # Feature columns
HR-Employee-Attrition.csv

👨‍💻 Author

Harmandeep Singh  
GitHub: https://github.com/Hrmn911/HR-attrition-analysis.git
