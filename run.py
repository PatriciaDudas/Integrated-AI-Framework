import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the data with specific date ranges to ensure we capture anomalies
def load_data_with_anomalies(filepath):
    print("Loading data with focus on anomaly periods...")
    
    # Define failure periods from the dataset documentation
    failure_periods = [
        ('2020-04-18 00:00:00', '2020-04-18 23:59:59'),  # #1
        ('2020-05-29 23:30:00', '2020-05-30 06:00:00'),  # #1 (second occurrence)
        ('2020-06-05 10:00:00', '2020-06-07 14:30:00'),  # #3
        ('2020-07-15 14:30:00', '2020-07-15 19:00:00')   # #4
    ]
    
    # Read in chunks to handle large file
    chunk_size = 100000
    chunks = []
    anomaly_chunks = []
    normal_chunks = []
    
    # Read the full file in chunks
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Convert timestamp to datetime
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
        
        # Check if this chunk contains any failure periods
        contains_anomaly = False
        for start, end in failure_periods:
            start_dt = pd.Timestamp(start)
            end_dt = pd.Timestamp(end)
            
            # If any timestamp in chunk falls within a failure period
            if ((chunk['timestamp'] >= start_dt) & (chunk['timestamp'] <= end_dt)).any():
                contains_anomaly = True
                
                # Filter to just the rows that fall in the failure period
                anomaly_rows = chunk[(chunk['timestamp'] >= start_dt) & (chunk['timestamp'] <= end_dt)]
                anomaly_chunks.append(anomaly_rows)
                
                # Also keep some context (normal data) from before and after the anomaly
                # We'll keep rows from 24 hours before to 24 hours after
                padding_before = chunk[(chunk['timestamp'] >= (start_dt - pd.Timedelta(hours=24))) & 
                                      (chunk['timestamp'] < start_dt)]
                padding_after = chunk[(chunk['timestamp'] > end_dt) & 
                                     (chunk['timestamp'] <= (end_dt + pd.Timedelta(hours=24)))]
                
                if not padding_before.empty:
                    normal_chunks.append(padding_before)
                if not padding_after.empty:
                    normal_chunks.append(padding_after)
                
                break
        
        # If no anomalies in this chunk, we'll randomly sample some rows to keep
        if not contains_anomaly:
            # Keep 5% of normal data for context
            if len(chunk) > 1000:
                normal_sample = chunk.sample(n=min(1000, len(chunk)), random_state=42)
                normal_chunks.append(normal_sample)
    
    # Combine all the data
    print(f"Found {len(anomaly_chunks)} chunks with anomalies")
    
    # Combine anomaly data
    if anomaly_chunks:
        anomaly_df = pd.concat(anomaly_chunks)
        print(f"Anomaly data: {len(anomaly_df)} rows")
    else:
        anomaly_df = pd.DataFrame()
        print("No anomaly data found!")
    
    # Combine normal data
    if normal_chunks:
        normal_df = pd.concat(normal_chunks)
        print(f"Normal data: {len(normal_df)} rows")
    else:
        normal_df = pd.DataFrame()
        print("No normal data found!")
    
    # Combine all data
    df = pd.concat([anomaly_df, normal_df])
    df = df.drop_duplicates()  # Remove any duplicates
    df = df.sort_values('timestamp')  # Sort by timestamp
    
    print(f"Final dataset: {len(df)} rows and {df.shape[1]} columns")
    return df

# Basic feature engineering
def basic_feature_engineering(df):
    print("Performing basic feature engineering...")
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Extract key relationships
    # Pressure differences
    df['TP3_TP2_diff'] = df['TP3'] - df['TP2']
    df['TP3_Reservoirs_diff'] = df['TP3'] - df['Reservoirs']
    
    # Create system state indicators based on Motor Current
    df['is_off'] = (df['Motor_current'] < 0.5).astype(int)
    df['is_offloaded'] = ((df['Motor_current'] >= 1.5) & (df['Motor_current'] <= 5.0)).astype(int)
    df['is_under_load'] = (df['Motor_current'] > 5.0).astype(int)
    
    # Operating mode based on COMP and DV_eletric
    df['load_mode'] = ((df['COMP'] == 0) & (df['DV_eletric'] == 1)).astype(int)
    
    print(f"Feature engineering complete. New shape: {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Create labels for anomalies
def create_labels(df):
    print("Creating labels for anomalies...")
    
    # Initialize labels to 0 (normal)
    df['anomaly'] = 0
    
    # Define failure periods from the dataset documentation
    failure_periods = [
        ('2020-04-18 00:00:00', '2020-04-18 23:59:59'),  # #1
        ('2020-05-29 23:30:00', '2020-05-30 06:00:00'),  # #1 (second occurrence)
        ('2020-06-05 10:00:00', '2020-06-07 14:30:00'),  # #3
        ('2020-07-15 14:30:00', '2020-07-15 19:00:00')   # #4
    ]
    
    # Mark anomalies with 1
    for start, end in failure_periods:
        mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
        df.loc[mask, 'anomaly'] = 1
    
    # Count the class distribution
    anomaly_count = df['anomaly'].sum()
    total_count = len(df)
    print(f"Anomaly instances: {anomaly_count} ({(anomaly_count/total_count)*100:.2f}%)")
    print(f"Normal instances: {total_count - anomaly_count} ({((total_count-anomaly_count)/total_count)*100:.2f}%)")
    
    return df

# Create RUL labels
def create_simple_rul_labels(df):
    print("Creating simple RUL labels...")
    
    # Initialize RUL column (in days)
    df['rul'] = 30  # Default to 30 days
    
    # Define failure times
    failure_times = [
        pd.Timestamp('2020-04-18 23:59:59'),
        pd.Timestamp('2020-05-30 06:00:00'),
        pd.Timestamp('2020-06-07 14:30:00'),
        pd.Timestamp('2020-07-15 19:00:00')
    ]
    
    # For data efficiency, only calculate RUL for each day, not every row
    # Create a date column
    df['date'] = df['timestamp'].dt.date
    
    # Get unique dates
    unique_dates = df['date'].unique()
    
    # Create a mapping from date to RUL
    date_to_rul = {}
    
    # For each date, calculate days until next failure
    for date in unique_dates:
        # Convert to timestamp for comparison
        date_timestamp = pd.Timestamp(date)
        
        # Find future failures
        future_failures = [ft for ft in failure_times if ft.date() > date]
        
        if future_failures:
            # Get the nearest future failure
            nearest_failure = min(future_failures)
            
            # Calculate RUL in days
            rul_days = (nearest_failure.date() - date).days
            
            # Cap at 30 days
            date_to_rul[date] = min(rul_days, 30)
        else:
            # If no future failures, set to max RUL
            date_to_rul[date] = 30
    
    # Apply the mapping to the DataFrame
    df['rul'] = df['date'].map(date_to_rul)
    
    # Drop the date column
    df = df.drop('date', axis=1)
    
    print(f"RUL statistics: Min={df['rul'].min():.2f}, Max={df['rul'].max():.2f}, Mean={df['rul'].mean():.2f}")
    
    return df

# Simplified data preprocessing
def simple_preprocessing(df, target='anomaly', sample_size=50000):
    print(f"Preprocessing data for {target} prediction...")
    
    # Sample data for faster processing
    if len(df) > sample_size:
        # If anomaly detection, ensure we keep all anomaly instances
        if target == 'anomaly':
            # Get all anomaly instances
            anomaly_df = df[df['anomaly'] == 1]
            sample_size_normal = min(sample_size - len(anomaly_df), len(df[df['anomaly'] == 0]))
            
            # Sample from normal instances
            normal_df = df[df['anomaly'] == 0].sample(sample_size_normal, random_state=42)
            
            # Combine and shuffle
            df_sampled = pd.concat([anomaly_df, normal_df]).sample(frac=1, random_state=42)
        else:
            # For RUL, stratify by RUL ranges
            df['rul_bucket'] = pd.cut(df['rul'], bins=5)
            df_sampled = df.groupby('rul_bucket').apply(
                lambda x: x.sample(min(len(x), sample_size // 5), random_state=42)
            ).reset_index(drop=True)
            df_sampled = df_sampled.drop('rul_bucket', axis=1)
    else:
        df_sampled = df
    
    # Select features - dropping unnecessary columns
    drop_cols = ['Unnamed: 0', 'timestamp', 'anomaly', 'rul']
    features = df_sampled.drop(drop_cols, axis=1, errors='ignore')
    
    # Create the target variable
    if target == 'anomaly':
        y = df_sampled['anomaly']
    else:  # 'rul'
        y = df_sampled['rul']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, 
        stratify=y if target == 'anomaly' else None
    )
    
    print(f"Final training set: {X_train.shape}, test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

# Train a lightweight anomaly detection model
def train_simple_anomaly_model(X_train, y_train):
    print("Training lightweight anomaly detection model...")
    
    # Get class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(f"Training class distribution: {class_distribution}")
    
    # Check if we have at least two classes
    if len(unique) < 2:
        print("ERROR: Only one class found in training data. Cannot train classifier.")
        print("Attempting to create a dummy model...")
        
        # Create a dummy model that always predicts the majority class
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='constant', constant=unique[0])
        model.fit(X_train, y_train)
        return model
    
    # Use Logistic Regression for faster training
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Anomaly detection model trained")
    
    return model

# Train a lightweight RUL model
def train_simple_rul_model(X_train, y_train):
    print("Training lightweight RUL model...")
    
    # Use XGBoost with limited depth/trees for faster training
    model = XGBRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    print("RUL model trained")
    
    return model

# Evaluate anomaly detection model
def evaluate_anomaly_model(model, X_test, y_test):
    print("Evaluating anomaly detection model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Anomaly Detection')
    plt.savefig('anomaly_confusion_matrix_simple.png', dpi=300, bbox_inches='tight')
    
    # Get ROC curve and AUC
    if hasattr(model, 'predict_proba'):
        from sklearn.metrics import roc_curve, auc
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Anomaly Detection')
        plt.legend(loc="lower right")
        plt.savefig('anomaly_roc_curve_simple.png', dpi=300, bbox_inches='tight')
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Evaluate RUL model
def evaluate_rul_model(model, X_test, y_test):
    print("Evaluating RUL prediction model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate custom accuracy for RUL
    # Consider a prediction within 20% of the actual RUL as accurate
    y_test_array = np.array(y_test)
    tolerance = 0.1 * y_test_array  # 20% tolerance
    
    # For near-zero RUL values, use a small fixed tolerance
    min_tolerance = 0.5  # half a day
    tolerance = np.maximum(tolerance, min_tolerance)
    
    accurate_predictions = np.abs(y_pred - y_test_array) <= tolerance
    rul_accuracy = np.mean(accurate_predictions)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RUL Accuracy (±10%): {rul_accuracy:.4f}")
    
    # Plot actual vs predicted RUL
    plt.figure(figsize=(10, 6))
    
    # Sample 1000 points for visibility
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(np.array(y_test)[indices], y_pred[indices], alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.xlabel('Actual RUL (days)')
    plt.ylabel('Predicted RUL (days)')
    plt.title('Actual vs Predicted RUL')
    plt.savefig('rul_prediction_simple.png', dpi=300, bbox_inches='tight')
    
    # Plot feature importances if applicable
    if hasattr(model, 'feature_importances_'):
        # Get top 10 feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 10 Feature Importances - RUL Prediction')
        plt.bar(range(10), importances[indices])
        plt.xticks(range(10), indices)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.savefig('rul_feature_importance_simple.png', dpi=300, bbox_inches='tight')
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'accuracy': rul_accuracy
    }

# Main function
def main(data_path):
    # Load data with focus on anomaly periods
    df = load_data_with_anomalies(data_path)
    
    # Perform basic feature engineering
    df = basic_feature_engineering(df)
    
    # Create anomaly labels
    df = create_labels(df)
    
    # Create RUL labels
    df = create_simple_rul_labels(df)
    
    # Check if we have anomaly data
    anomaly_count = df['anomaly'].sum()
    if anomaly_count == 0:
        print("ERROR: No anomalies found in the dataset. Cannot train anomaly detection model.")
        return None, None, None, None
    
    print(f"Found {anomaly_count} anomaly instances ({(anomaly_count/len(df))*100:.2f}% of data)")
    
    # Anomaly Detection Model
    print("\n===== ANOMALY DETECTION MODEL =====")
    X_train_anomaly, X_test_anomaly, y_train_anomaly, y_test_anomaly, scaler_anomaly = simple_preprocessing(
        df, 
        target='anomaly',
        sample_size=50000
    )
    
    # Check if test set has both classes
    test_classes = np.unique(y_test_anomaly)
    if len(test_classes) < 2:
        print(f"WARNING: Test set contains only class {test_classes[0]}. Evaluation will be limited.")
    
    anomaly_model = train_simple_anomaly_model(X_train_anomaly, y_train_anomaly)
    anomaly_metrics = evaluate_anomaly_model(anomaly_model, X_test_anomaly, y_test_anomaly)
    
    # RUL Prediction Model
    print("\n===== RUL PREDICTION MODEL =====")
    X_train_rul, X_test_rul, y_train_rul, y_test_rul, scaler_rul = simple_preprocessing(
        df, 
        target='rul',
        sample_size=50000
    )
    
    rul_model = train_simple_rul_model(X_train_rul, y_train_rul)
    rul_metrics = evaluate_rul_model(rul_model, X_test_rul, y_test_rul)
    
    # Visualize sensor patterns during normal and anomaly periods
    print("\n===== CREATING VISUALIZATIONS =====")
    
    # Check if we have anomaly data for visualization
    if anomaly_count > 0:
        # Create a visualization showing normal vs. anomaly patterns
        anomaly_df = df[df['anomaly'] == 1].sample(min(1000, df['anomaly'].sum()), random_state=42)
        normal_df = df[df['anomaly'] == 0].sample(min(1000, (df['anomaly'] == 0).sum()), random_state=42)
        
        # Plot key sensors
        key_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Motor_current', 'Oil_temperature']
        
        fig, axes = plt.subplots(len(key_sensors), 1, figsize=(12, 18), sharex=True)
        
        for i, sensor in enumerate(key_sensors):
            axes[i].hist(normal_df[sensor], bins=30, alpha=0.5, label='Normal', color='green')
            axes[i].hist(anomaly_df[sensor], bins=30, alpha=0.5, label='Anomaly', color='red')
            axes[i].set_title(f'{sensor} Distribution')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
        
        plt.xlabel('Sensor Value')
        plt.tight_layout()
        plt.savefig('sensor_distributions_simple.png', dpi=300, bbox_inches='tight')
    else:
        print("Skipping normal vs. anomaly visualization - no anomaly data available")
    
    # Plot RUL over time with anomalies highlighted
    plt.figure(figsize=(12, 6))
    
    # Sample for better visualization
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    sample_df = sample_df.sort_values('timestamp')
    
    # Plot RUL
    plt.scatter(
        sample_df['timestamp'], 
        sample_df['rul'], 
        c=sample_df['anomaly'],
        cmap='coolwarm',
        alpha=0.7,
        s=15
    )
    
    plt.colorbar(label='Anomaly (1) / Normal (0)')
    plt.xlabel('Time')
    plt.ylabel('RUL (days)')
    plt.title('RUL Over Time with Anomalies Highlighted')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rul_time_anomalies.png', dpi=300, bbox_inches='tight')
    
    # Print summary
    print("\n===== MODEL PERFORMANCE SUMMARY =====")
    if anomaly_metrics:
        print("Anomaly Detection:")
        print(f"Accuracy: {anomaly_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"Balanced Accuracy: {anomaly_metrics.get('balanced_accuracy', 'N/A'):.4f}")
        print(f"F1 Score: {anomaly_metrics.get('f1', 'N/A'):.4f}")
    else:
        print("Anomaly Detection: Could not train model")
        
    print("\nRUL Prediction:")
    print(f"Accuracy (±10%): {rul_metrics.get('accuracy', 'N/A'):.4f}")
    print(f"RMSE: {rul_metrics.get('rmse', 'N/A'):.4f}")
    print(f"R² Score: {rul_metrics.get('r2', 'N/A'):.4f}")
    
    return anomaly_model, rul_model, anomaly_metrics, rul_metrics

if __name__ == "__main__":
    # Replace with your actual path
    data_path = "MetroPT3.csv"
    
    try:
        # Run the main function
        anomaly_model, rul_model, anomaly_metrics, rul_metrics = main(data_path)
        
        if anomaly_model is None:
            print("\nWARNING: The program could not find any anomaly instances in the dataset.")
            print("Possible causes:")
            print("1. The dataset may not include the time periods with known failures")
            print("2. The timestamp format in the dataset may be different from what's expected")
            print("\nTry running the program with the full dataset or checking the timestamp format.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nSuggestions to fix the error:")
        print("1. Make sure the CSV file exists and has the expected columns")
        print("2. Check if the timestamp column has a consistent datetime format")
        print("3. Verify that the dataset includes data from April to July 2020 (when anomalies occur)")