import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Data loading
df = pd.read_csv("HR-Employee-Attrition.csv")
print(df)

# Data info gathering

# df.shape
# df.info()
# df.describe()

# Data cleaning

print(df.isnull().sum())
print(df.duplicated().sum())

# Data Visulization

# Countplot
sns.countplot(x="Attrition", data=df)

plt.title("Employee Attrition Distribution")

plt.show()

sns.countplot(
    x="OverTime",
    hue="Attrition",
    data=df
)

plt.show()


sns.countplot(
    x="Department",
    hue="Attrition",
    data=df
)

plt.xticks(rotation=45)

plt.show()


# Boxplot
sns.boxplot(y="Age", data=df)

plt.show()

sns.boxplot(y="Age", data=df)

plt.show()

sns.boxplot(y="YearsAtCompany", data=df)

plt.show()

# Scatterplot

sns.scatterplot(
    x="Age",
    y="MonthlyIncome",
    hue="Attrition",
    data=df
)

plt.show()


# Checking Outliers

Q1 = df["MonthlyIncome"].quantile(0.25)
Q3 = df["MonthlyIncome"].quantile(0.75)

IQR = Q3 - Q1

outliers = df[
    (df["MonthlyIncome"] < Q1 - 1.5 * IQR) |
    (df["MonthlyIncome"] > Q3 + 1.5 * IQR)
]

print("Number of outliers:", len(outliers))


# Remove useless columns

df.drop(
    ["EmployeeCount",
     "EmployeeNumber",
     "Over18",
     "StandardHours"],
    axis=1,
    inplace=True
)

# Encoding

df["Attrition"] = df["Attrition"].map({
    "No":0,
    "Yes":1
})

print(df.select_dtypes(include="object").columns)

df = pd.get_dummies(
    df,
    drop_first=True
)
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

joblib.dump(X.columns, "features.pkl")

# Heatmap

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.show()

# Split X and Y

X = df.drop("Attrition", axis=1)

y = df["Attrition"]

# Train  and Test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Logistic Regression

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save feature columns 
joblib.dump(X.columns, "features.pkl")

print("Model, scaler and features saved successfully!")
print("Model Saved Successfully!")

# Prediction

y_pred = model.predict(X_test)

# Accuray

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

# Classifiaction report

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

y_prob = model.predict_proba(X_test)[:, 1]

print(
    "ROC AUC Score:",
    roc_auc_score(y_test, y_prob)
)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

print(importance.head(15))


importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(
    by="Coefficient"
)

print(importance.head(15))


for i in range(10):
    print("Pred:", y_pred[i])
    print("Actual:", y_test.iloc[i])
    print("----")


