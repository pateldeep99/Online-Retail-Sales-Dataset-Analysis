# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    silhouette_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
# import statsmodels.api as sm
# from statsmodels.stats.outliers_influence import variance_inflation_factor


# Function to clean data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df


def train_regression_model(df):
    df['month'] = df['timestamp'].dt.month
    df_sales = df.groupby(['month']).agg({'total_amount': 'sum', 'quantity': 'sum', 'discount': 'mean'}).reset_index()
    X = df_sales[['month', 'quantity', 'discount']]
    y = df_sales['total_amount']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return lin_reg, mse, r2, y_test, y_pred, X


def train_classification_model(df):
    customer_data = df.groupby('customer_id').agg({'total_amount': 'sum', 'timestamp': 'max'})
    last_date = df['timestamp'].max()
    customer_data['Churn'] = (last_date - customer_data['timestamp']).dt.days > 90
    X = customer_data[['total_amount']]
    y = customer_data['Churn'].astype(int)

    if y.nunique() < 2:
        print("Error: Insufficient class diversity.")
        return None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return model, accuracy, precision, recall, f1, y_test, y_pred, y_proba


def train_decision_tree_classifier(df):
    customer_data = df.groupby('customer_id').agg({'total_amount': 'sum', 'timestamp': 'max'})
    last_date = df['timestamp'].max()
    customer_data['Churn'] = (last_date - customer_data['timestamp']).dt.days > 90
    X = customer_data[['total_amount']]
    y = customer_data['Churn'].astype(int)

    dt_X_train, dt_X_test, dt_y_train, dt_y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(dt_X_train, dt_y_train)
    dt_y_pred = dt_model.predict(dt_X_test)
    dt_y_proba = dt_model.predict_proba(dt_X_test)  # Get probabilities for ROC curve

    dt_accuracy = accuracy_score(dt_y_test, dt_y_pred)
    dt_precision = precision_score(dt_y_test, dt_y_pred)
    dt_recall = recall_score(dt_y_test, dt_y_pred)
    dt_f1 = f1_score(dt_y_test, dt_y_pred)

    return dt_model, dt_accuracy, dt_precision, dt_recall, dt_f1, dt_y_test, dt_y_pred, dt_y_proba


def train_clustering_model(df, k=3):
    customer_data = df.groupby('customer_id').agg({'total_amount': 'sum', 'quantity': 'sum'})
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(customer_data)
    customer_data['Cluster'] = clusters

    silhouette = silhouette_score(customer_data[['total_amount', 'quantity']], clusters)
    inertia = kmeans.inertia_

    return kmeans, silhouette, inertia, customer_data
