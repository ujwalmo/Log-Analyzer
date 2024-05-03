import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    # Encode categorical features
    label_encoders = {}
    for column in ['IP','URL', 'Status','TimeStamp']:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders

def check_anomalies(group):
    anomalies = []
    prev_time = None
    time_threshold = timedelta(minutes=20)

    for index, row in group.iterrows():
        if prev_time is not None and row['TimeStamp'] - prev_time <= time_threshold:
            if row['Status'] in [401, 403, 400]:
                anomalies.append(1)
            else:
                anomalies.append(0)
        else:
            anomalies.append(0)
        prev_time = row['TimeStamp']

    return anomalies

def detect_anomalies_svm(df, svm_model):
    X = df[['IP', 'URL', 'Status', 'TimeStamp']]
    df['SVM_Anomaly'] = (svm_model.predict(X) == 1).astype(int)
    
    return df

def detect_anomalies_iforest(df, iforest_model):
    X = df[['IP','URL', 'Status','TimeStamp']]
    df['IForest_Anomaly'] = (iforest_model.predict(X) == -1).astype(int)
    
    return df

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


def main():
    st.title('Anomaly Detection Dashboard')
    # File uploader for CSV file
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    # Dropdown for selecting model
    model_selection = st.selectbox("Select Model", ["None", "SVM", "Isolation Forest"])

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

        # Calculate 'Anomaly' column using check_anomalies function
        df['Anomaly'] = df.groupby('IP').apply(check_anomalies).explode().reset_index(drop=True)
        df.dropna(subset=['Anomaly'], inplace=True)

        # Preprocess data and keep label encoders
        df, label_encoders = preprocess_data(df)

        if model_selection != "None":
            # Load pre-trained models if model is selected
            if model_selection == "SVM":
                svm_model = joblib.load('svm_model.pkl')
                df = detect_anomalies_svm(df, svm_model)
            elif model_selection == "Isolation Forest":
                iforest_model = joblib.load('isolation_forest_model.pkl')
                df = detect_anomalies_iforest(df, iforest_model)

        # Decode URL and Status columns
        df['IP'] = label_encoders['IP'].inverse_transform(df['IP'])
        df['TimeStamp'] = label_encoders['TimeStamp'].inverse_transform(df['TimeStamp'])
        df['URL'] = label_encoders['URL'].inverse_transform(df['URL'])
        df['Status'] = label_encoders['Status'].inverse_transform(df['Status'])

        # Display DataFrame after adding predicted values
        st.subheader("DataFrame with Predicted Anomalies")
        st.table(df.head(10))

        if 'Anomaly' in df.columns:
            df['Anomaly'] = df['Anomaly'].astype(int)
            if 'SVM_Anomaly' in df.columns:
                df['SVM_Anomaly'] = df['SVM_Anomaly'].astype(int)
            if 'IForest_Anomaly' in df.columns:
                df['IForest_Anomaly'] = df['IForest_Anomaly'].astype(int)
                
            if model_selection != "None":
                # Confusion matrix
                st.subheader('Confusion Matrix')
                if model_selection == "SVM":
                    if 'SVM_Anomaly' in df.columns:
                        cm = confusion_matrix(df['Anomaly'], df['SVM_Anomaly'])
                        plot_confusion_matrix(cm, labels=['Normal', 'Anomaly'])
                        
                        # Calculate accuracy
                        accuracy = (df['SVM_Anomaly'] == df['Anomaly']).mean()
                        st.write("One-Class SVM Accuracy:", accuracy)
                        
                elif model_selection == "Isolation Forest":
                    if 'IForest_Anomaly' in df.columns:
                        cm = confusion_matrix(df['Anomaly'], df['IForest_Anomaly'])
                        plot_confusion_matrix(cm, labels=['Normal', 'Anomaly'])
                        
                        # Calculate accuracy
                        accuracy = (df['IForest_Anomaly'] == df['Anomaly']).mean()
                        st.write("Isolation Forest Accuracy:", accuracy)

                # ROC Curve
                st.subheader('ROC Curve')
                if model_selection == "SVM":
                    if 'SVM_Anomaly' in df.columns:
                        fpr, tpr, thresholds = roc_curve(df['Anomaly'], df['SVM_Anomaly'])
                        st.line_chart(pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr}))
                elif model_selection == "Isolation Forest":
                    if 'IForest_Anomaly' in df.columns:
                        fpr, tpr, thresholds = roc_curve(df['Anomaly'], df['IForest_Anomaly'])
                        st.line_chart(pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr}))

if __name__ == '__main__':
    main()