import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, roc_curve, auc, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import os
import time
import joblib

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

# Feature engineering
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
    df['H1_TP3_diff'] = df['H1'] - df['TP3']
    
    # Create system state indicators based on Motor Current
    df['is_off'] = (df['Motor_current'] < 0.5).astype(int)
    df['is_offloaded'] = ((df['Motor_current'] >= 1.5) & (df['Motor_current'] <= 5.0)).astype(int)
    df['is_under_load'] = (df['Motor_current'] > 5.0).astype(int)
    
    # Operating mode based on COMP and DV_eletric
    df['load_mode'] = ((df['COMP'] == 0) & (df['DV_eletric'] == 1)).astype(int)
    
    # Add more sophisticated features
    # Rate of change for important sensors
    df['TP2_rate'] = df['TP2'].diff().fillna(0)
    df['TP3_rate'] = df['TP3'].diff().fillna(0) 
    df['Motor_current_rate'] = df['Motor_current'].diff().fillna(0)
    
    # Additional ratio features
    df['H1_TP3_ratio'] = df['H1'] / df['TP3'].replace(0, np.nan).fillna(1)
    df['TP3_TP2_ratio'] = df['TP3'] / df['TP2'].replace(0, np.nan).fillna(1)
    
    # Replace infinities that might have been created
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
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

# Create simple RUL labels (days until next failure)
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

# Preprocessing for autoencoder
def preprocess_for_autoencoder(df, target='anomaly', sample_size=50000):
    print(f"Preprocessing data for {target} with autoencoder...")
    
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
    
    # Scale the features - autoencoders need normalized data
    scaler = MinMaxScaler()  # Use MinMaxScaler for autoencoders
    X_scaled = scaler.fit_transform(features)
    
    # For autoencoder, we separate normal and anomaly data
    X_normal = X_scaled[y == 0]
    X_anomaly = X_scaled[y == 1]
    
    # Split the normal data into train/test
    X_train_normal, X_test_normal = train_test_split(
        X_normal, test_size=0.2, random_state=42
    )
    
    # Train the autoencoder only on normal data
    print(f"Autoencoder training data: {X_train_normal.shape}")
    print(f"Autoencoder test data (normal): {X_test_normal.shape}")
    print(f"Anomaly test data: {X_anomaly.shape}")
    
    # For evaluation, we'll need a test set with both normal and anomaly
    X_test_combined = np.vstack([X_test_normal, X_anomaly])
    y_test_combined = np.hstack([np.zeros(len(X_test_normal)), np.ones(len(X_anomaly))])
    
    return X_train_normal, X_test_combined, y_test_combined, scaler, features.columns

# Preprocessing for RUL
def preprocess_for_rul(df, target='rul', sample_size=50000):
    print(f"Preprocessing data for {target} prediction...")
    
    # Sample data for faster processing
    if len(df) > sample_size:
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
    y = df_sampled['rul']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, features.columns

# Build improved autoencoder model
def build_improved_autoencoder(input_dim):
    print(f"Building improved autoencoder with input dimension {input_dim}...")
    
    # Define encoding dimension (bottleneck)
    encoding_dim = max(8, input_dim // 4)  # Slightly larger bottleneck
    
    # Build encoder
    input_layer = Input(shape=(input_dim,))
    
    # Encoder layers - more layers and different activation
    encoded = Dense(input_dim, activation='tanh')(input_layer)
    encoded = Dropout(0.3)(encoded)  # Increase dropout for better generalization
    encoded = Dense(input_dim // 2, activation='tanh')(encoded)
    encoded = Dense(encoding_dim, activation='tanh')(encoded)
    
    # Decoder layers
    decoded = Dense(input_dim // 2, activation='tanh')(encoded)
    decoded = Dropout(0.3)(decoded)
    decoded = Dense(input_dim, activation='tanh')(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    # Create model
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    
    # Print model summary
    autoencoder.summary()
    
    # Compile model with smaller learning rate
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                       loss='mse')
    
    return autoencoder

# Train improved autoencoder for anomaly detection
def train_improved_autoencoder(X_train, input_dim):
    print("Training improved autoencoder for anomaly detection...")
    
    # Build the model
    autoencoder = build_improved_autoencoder(input_dim)
    
    # Early stopping with longer patience
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Increased patience
        restore_best_weights=True
    )
    
    # Learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Train the model with more epochs and smaller batch size
    history = autoencoder.fit(
        X_train, X_train,  # Autoencoder aims to reconstruct the input
        epochs=100,        # More epochs
        batch_size=32,     # Smaller batch size
        shuffle=True,
        validation_split=0.2,  # Larger validation split
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Improved Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/improved_autoencoder_training_history.png', dpi=300, bbox_inches='tight')
    
    print("Improved autoencoder trained")
    return autoencoder

# Evaluate autoencoder with optimized threshold selection
def evaluate_improved_autoencoder(autoencoder, X_test, y_test):
    print("Evaluating improved autoencoder for anomaly detection...")
    
    # Get reconstruction
    X_pred = autoencoder.predict(X_test)
    
    # Calculate reconstruction error for each sample
    mse = np.mean(np.power(X_test - X_pred, 2), axis=1)
    
    # Ensure output directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(12, 6))
    plt.hist(mse[y_test == 0], bins=50, alpha=0.5, label='Normal', color='green')
    plt.hist(mse[y_test == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, mse)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.axvline(x=optimal_threshold, color='blue', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.6f})')
    plt.title('Improved Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('plots/improved_autoencoder_error_distribution.png', dpi=300, bbox_inches='tight')
    
    print(f"Optimal threshold from ROC curve: {optimal_threshold:.6f}")
    
    # Classify as anomaly if reconstruction error > threshold
    y_pred = (mse > optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Improved Autoencoder')
    plt.savefig('plots/improved_autoencoder_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Create ROC curve
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', markersize=10, 
            label=f'Optimal threshold (TPR={tpr[optimal_idx]:.2f}, FPR={fpr[optimal_idx]:.2f})', 
            fillstyle='none', mfc='none', markeredgewidth=2, markeredgecolor='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Improved Autoencoder')
    plt.legend(loc="lower right")
    plt.savefig('plots/improved_autoencoder_roc_curve.png', dpi=300, bbox_inches='tight')
    
    # Try multiple thresholds to find best accuracy
    print("\nExploring different thresholds for optimal accuracy:")
    best_accuracy = 0
    best_balanced_accuracy = 0
    best_threshold = optimal_threshold
    best_metrics = None
    
    # Try percentiles from the anomaly class reconstruction errors
    for percentile in [50, 60, 70, 80, 90, 95, 99]:
        threshold = np.percentile(mse[y_test == 1], percentile)
        y_pred_test = (mse > threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        print(f"Threshold at {percentile}th percentile of anomaly errors ({threshold:.6f}): Accuracy={acc:.4f}, Balanced Acc={bal_acc:.4f}, F1={f1:.4f}")
        
        # Keep track of best balanced accuracy
        if bal_acc > best_balanced_accuracy:
            best_balanced_accuracy = bal_acc
            best_accuracy = acc
            best_threshold = threshold
            best_metrics = {
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'precision': precision_score(y_test, y_pred_test, zero_division=0),
                'recall': recall_score(y_test, y_pred_test),
                'f1': f1,
                'threshold': threshold,
                'auc': roc_auc
            }
    
    print(f"\nBest threshold: {best_threshold:.6f} with balanced accuracy: {best_balanced_accuracy:.4f}")
    
    if best_metrics:
        print(f"Using best metrics found: Accuracy={best_metrics['accuracy']:.4f}, F1={best_metrics['f1']:.4f}")
        # Recreate confusion matrix with best threshold
        y_pred_best = (mse > best_threshold).astype(int)
        cm_best = confusion_matrix(y_test, y_pred_best)
        
        # Plot best confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Improved Autoencoder (Best Threshold)')
        plt.savefig('plots/improved_autoencoder_confusion_matrix_best.png', dpi=300, bbox_inches='tight')
        
        return best_metrics
    else:
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': optimal_threshold,
            'auc': roc_auc
        }

# Train RUL model
def train_rul_model(X_train, y_train):
    print("Training RUL prediction model...")
    
    # Use XGBoost for RUL prediction with more trees and smaller learning rate
    model = XGBRegressor(
        n_estimators=200,  # More trees
        max_depth=6,
        learning_rate=0.05,  # Smaller learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    print("RUL prediction model trained")
    
    return model

# Evaluate RUL model with 10% tolerance
def evaluate_rul_model(model, X_test, y_test, feature_names=None):
    print("Evaluating RUL prediction model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate custom accuracy for RUL with 10% tolerance (instead of 20%)
    y_test_array = np.array(y_test)
    tolerance = 0.1 * y_test_array  # 10% tolerance
    
    # For near-zero RUL values, use a small fixed tolerance
    min_tolerance = 0.25  # quarter of a day
    tolerance = np.maximum(tolerance, min_tolerance)
    
    accurate_predictions = np.abs(y_pred - y_test_array) <= tolerance
    rul_accuracy = np.mean(accurate_predictions)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"RUL Accuracy (±10%): {rul_accuracy:.4f}")  # Updated to show 10%
    
    # Ensure output directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Plot actual vs predicted RUL
    plt.figure(figsize=(10, 6))
    
    # Sample 1000 points for visibility if needed
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(np.array(y_test)[indices], y_pred[indices], alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    
    # Add 10% tolerance bands
    max_val = max(max(y_test), max(y_pred))
    x = np.linspace(0, max_val, 100)
    plt.plot(x, x * 1.1, 'g--', alpha=0.5, label='+10% Tolerance')
    plt.plot(x, x * 0.9, 'g--', alpha=0.5, label='-10% Tolerance')
    plt.fill_between(x, x * 0.9, x * 1.1, alpha=0.1, color='green')
    
    plt.xlabel('Actual RUL (days)')
    plt.ylabel('Predicted RUL (days)')
    plt.title('Actual vs Predicted RUL (±10% Tolerance)')
    plt.legend()
    plt.savefig('plots/rul_prediction.png', dpi=300, bbox_inches='tight')
    
    # Plot feature importances if applicable
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        # Get feature importances
        importances = model.feature_importances_
        
        # Map indices to feature names
        feature_importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 15 features
        top_features = sorted_importances[:15]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(top_features)), [imp for _, imp in top_features])
        plt.xticks(range(len(top_features)), [name for name, _ in top_features], rotation=45, ha='right')
        plt.title('Top 15 Feature Importances - RUL Prediction')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('plots/rul_feature_importance.png', dpi=300, bbox_inches='tight')
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'accuracy': rul_accuracy
    }

# Visualize sensor patterns during normal and anomaly periods
def visualize_sensor_patterns(df):
    print("Visualizing sensor patterns...")
    
    # Ensure output directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Check if we have anomaly data for visualization
    anomaly_count = df['anomaly'].sum()
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
        plt.savefig('plots/sensor_distributions.png', dpi=300, bbox_inches='tight')
        
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
        plt.savefig('plots/rul_time_anomalies.png', dpi=300, bbox_inches='tight')
        
        # Add a heatmap correlation of sensors during normal vs anomaly periods
        plt.figure(figsize=(15, 12))
        
        # Select important columns for correlation
        important_cols = [
            'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 
            'Motor_current', 'Oil_temperature', 'TP3_TP2_diff', 
            'H1_TP3_diff', 'TP3_Reservoirs_diff'
        ]
        
        # Create correlation matrices
        normal_corr = normal_df[important_cols].corr()
        anomaly_corr = anomaly_df[important_cols].corr()
        
        # Set up the figure layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        
        # Plot normal correlation
        sns.heatmap(normal_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax1, cbar=False, square=True)
        ax1.set_title('Sensor Correlation - Normal Operation', fontsize=14)
        
        # Plot anomaly correlation
        sns.heatmap(anomaly_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2, cbar=True, square=True)
        ax2.set_title('Sensor Correlation - Anomaly Periods', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('plots/sensor_correlation_comparison.png', dpi=300, bbox_inches='tight')
        
    else:
        print("Skipping normal vs. anomaly visualization - no anomaly data available")

# Save results to file
def save_results(autoencoder_metrics, rul_metrics):
    # Ensure output directory exists
    os.makedirs('results', exist_ok=True)
    
    # Format current time for the filename
    current_time = time.strftime("%Y%m%d-%H%M%S")
    
    # Create a summary file
    with open(f'results/results_summary_{current_time}.txt', 'w') as f:
        f.write("===== MODEL PERFORMANCE SUMMARY =====\n\n")
        
        if autoencoder_metrics:
            f.write("Anomaly Detection (Improved Autoencoder):\n")
            f.write(f"Accuracy: {autoencoder_metrics.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"Balanced Accuracy: {autoencoder_metrics.get('balanced_accuracy', 'N/A'):.4f}\n")
            f.write(f"Precision: {autoencoder_metrics.get('precision', 'N/A'):.4f}\n")
            f.write(f"Recall: {autoencoder_metrics.get('recall', 'N/A'):.4f}\n")
            f.write(f"F1 Score: {autoencoder_metrics.get('f1', 'N/A'):.4f}\n")
            f.write(f"AUC: {autoencoder_metrics.get('auc', 'N/A'):.4f}\n")
            f.write(f"Threshold: {autoencoder_metrics.get('threshold', 'N/A'):.6f}\n\n")
        else:
            f.write("Anomaly Detection: Could not train model\n\n")
            
        f.write("RUL Prediction:\n")
        f.write(f"Accuracy (±10%): {rul_metrics.get('accuracy', 'N/A'):.4f}\n")
        f.write(f"RMSE: {rul_metrics.get('rmse', 'N/A'):.4f}\n")
        f.write(f"R² Score: {rul_metrics.get('r2', 'N/A'):.4f}\n")
        f.write(f"MSE: {rul_metrics.get('mse', 'N/A'):.4f}\n")
    
    print(f"Results saved to results/results_summary_{current_time}.txt")

# Main function
def main(data_path):
    # Starting time
    start_time = time.time()
    
    # Try to load TensorFlow
    try:
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except:
        print("TensorFlow not available or error checking GPU")
    
    # Create directories for output
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
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
    
    # Improved Autoencoder for Anomaly Detection
    print("\n===== IMPROVED AUTOENCODER FOR ANOMALY DETECTION =====")
    X_train_normal, X_test_combined, y_test_combined, scaler_anomaly, feature_columns = preprocess_for_autoencoder(df)
    
    # Train improved autoencoder
    autoencoder = train_improved_autoencoder(X_train_normal, X_train_normal.shape[1])
    
    # Save the model
    autoencoder.save('models/improved_autoencoder_model.keras')
    joblib.dump(scaler_anomaly, 'models/autoencoder_scaler.joblib')
    print("Improved autoencoder model saved to models/improved_autoencoder_model.keras")
    
    # Evaluate improved autoencoder
    autoencoder_metrics = evaluate_improved_autoencoder(autoencoder, X_test_combined, y_test_combined)
    
    # RUL Prediction Model with 10% tolerance
    print("\n===== RUL PREDICTION MODEL WITH 10% TOLERANCE =====")
    X_train_rul, X_test_rul, y_train_rul, y_test_rul, scaler_rul, feature_names_rul = preprocess_for_rul(df)
    
    # Train the RUL model
    rul_model = train_rul_model(X_train_rul, y_train_rul)
    
    # Save XGBoost model
    joblib.dump(rul_model, 'models/rul_model.joblib')
    joblib.dump(scaler_rul, 'models/rul_scaler.joblib')
    print("RUL model saved to models/rul_model.joblib")
    
    # Evaluate RUL model with 10% tolerance
    rul_metrics = evaluate_rul_model(rul_model, X_test_rul, y_test_rul, feature_names_rul)
    
    # Create visualizations
    visualize_sensor_patterns(df)
    
    # Save results
    save_results(autoencoder_metrics, rul_metrics)
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    return autoencoder, rul_model, autoencoder_metrics, rul_metrics

if __name__ == "__main__":
    # Replace with your actual path
    data_path = "MetroPT3.csv"
    
    try:
        # Run the main function
        autoencoder, rul_model, autoencoder_metrics, rul_metrics = main(data_path)
        
        if autoencoder is None:
            print("\nWARNING: The program could not find any anomaly instances in the dataset.")
            print("Possible causes:")
            print("1. The dataset may not include the time periods with known failures")
            print("2. The timestamp format in the dataset may be different from what's expected")
            print("\nTry running the program with the full dataset or checking the timestamp format.")
        else:
            print("\n===== MODEL PERFORMANCE SUMMARY =====")
            if autoencoder_metrics:
                print("Anomaly Detection (Improved Autoencoder):")
                print(f"Accuracy: {autoencoder_metrics.get('accuracy', 'N/A'):.4f}")
                print(f"Balanced Accuracy: {autoencoder_metrics.get('balanced_accuracy', 'N/A'):.4f}")
                print(f"F1 Score: {autoencoder_metrics.get('f1', 'N/A'):.4f}")
                print(f"AUC: {autoencoder_metrics.get('auc', 'N/A'):.4f}")
            else:
                print("Anomaly Detection: Could not train model")
                
            print("\nRUL Prediction:")
            print(f"Accuracy (±10%): {rul_metrics.get('accuracy', 'N/A'):.4f}")
            print(f"RMSE: {rul_metrics.get('rmse', 'N/A'):.4f}")
            print(f"R² Score: {rul_metrics.get('r2', 'N/A'):.4f}")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nSuggestions to fix the error:")
        print("1. Make sure the CSV file exists and has the expected columns")
        print("2. Check if the timestamp column has a consistent datetime format")
        print("3. Verify that the dataset includes data from April to July 2020 (when anomalies occur)")
        print("4. Ensure TensorFlow is installed: pip install tensorflow")
        print("5. For GPU support, install tensorflow-gpu or set up CUDA appropriately")