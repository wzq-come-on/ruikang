import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
data = pd.read_csv('mushroom(1).csv')

# 数据预处理
def preprocess_data(data):
    # 对所有特征进行Label Encoding
    labelencoder = LabelEncoder()
    for column in data.columns:
        data[column] = labelencoder.fit_transform(data[column])
    return data

data = preprocess_data(data)

# 分离特征和标签
X = data.drop('class', axis=1)
y = data['class']

# 确认特征数量
print(f'Number of features after preprocessing: {X.shape[1]}')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型调参
def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

# 选择和训练模型
model = tune_model(X_train, y_train)

# 确认训练特征数量
print(f'Number of features used for training: {X_train.shape[1]}')

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# 特征重要性
importances = model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
print(feature_importances.sort_values(by='importance', ascending=False))

# 特征重要性可视化
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.sort_values(by='importance', ascending=False))
plt.title('Feature Importances')
plt.show()

# 保存模型
joblib.dump(model, 'mushroom_classifier.pkl')

# 加载模型并测试
loaded_model = joblib.load('mushroom_classifier.pkl')
loaded_model_pred = loaded_model.predict(X_test)

# 确认加载模型后的预测特征数量
print(f'Number of features used for prediction: {X_test.shape[1]}')

# 评估加载模型后的预测
loaded_accuracy = accuracy_score(y_test, loaded_model_pred)
print(f'Accuracy after loading model: {loaded_accuracy:.2f}')
