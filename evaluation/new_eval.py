#!/usr/bin/env python3
"""
NPrint Traffic Classifier

This script trains and evaluates Random Forest and Decision Tree models
to classify network traffic based on nprint files.

Features:
- Filters out unwanted columns (IPv6, IPv4 options, ICMP)
- Allows toggling IP and port features
- Controls number of packets/rows to use as features
- Memory-efficient batch processing
- Feature importance analysis for Random Forest
- Visualization of feature distributions
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train network traffic classifier using nprint files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Directory arguments
    parser.add_argument('--real-dir', required=True, 
                        help='Directory with real nprint files')
    parser.add_argument('--synthetic-dir', required=True, 
                        help='Directory with synthetic nprint files')
    parser.add_argument('--output-dir', default='output', 
                        help='Directory to save results and models')
    
    # Processing arguments
    parser.add_argument('--batch-size', type=int, default=50, 
                        help='Number of files to process in each batch')
    parser.add_argument('--max-rows', type=int, default=1024, 
                        help='Maximum number of rows/packets to use from each file')
    parser.add_argument('--include-ip-port', action='store_true', 
                        help='Include IP and port features (ipv4_src*, ipv4_dst*, tcp_*port*, udp_*port*)')
    
    # Model arguments
    parser.add_argument('--rf-trees', type=int, default=100, 
                        help='Number of trees for Random Forest')
    parser.add_argument('--test-size', type=float, default=0.3, 
                        help='Proportion of data for testing')
    parser.add_argument('--random-state', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    # Output control
    parser.add_argument('--visualize-top-n', type=int, default=5,
                        help='Number of top features to visualize')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def get_column_names(sample_file):
    """
    Get column names from a sample nprint file
    
    Args:
        sample_file (str): Path to an nprint file
        
    Returns:
        list: List of column names
    """
    with open(sample_file, 'r') as f:
        header_line = f.readline().strip()
        columns = header_line.split(',')
    return columns


def filter_columns(columns, include_ip_port=True):
    """
    Filter out unwanted columns and optionally include/exclude IP/port columns
    
    Args:
        columns (list): List of column names
        include_ip_port (bool): Whether to include IP/port columns
    
    Returns:
        tuple: (filtered_columns, column_indices)
            - filtered_columns: List of filtered column names
            - column_indices: List of indices for the filtered columns
    """
    # Patterns to exclude
    exclude_patterns = [
        r"src_ip",
        r"ipv6_.*",
        r"ipv4_opt_.*",
        r"rts",
        r"icmp_.*"
    ]
    
    # Patterns to conditionally include
    conditional_patterns = [
        r"ipv4_src.*",
        r"ipv4_dst.*",
        r"tcp_sprt_.*",
        r"tcp_dprt.*",
        r"udp_sport.*",
        r"udp_dport.*"
    ]
    
    # Combine exclude patterns into a single regex
    exclude_regex = re.compile("|".join(exclude_patterns))
    
    # Combine conditional patterns into a single regex
    conditional_regex = re.compile("|".join(conditional_patterns))
    
    filtered_columns = []
    column_indices = []
    
    for i, col in enumerate(columns):
        # Check if column should be excluded
        if exclude_regex.match(col):
            continue
        
        # Check if column is conditional and should be excluded
        if conditional_regex.match(col) and not include_ip_port:
            continue
        
        # Column passes all filters
        filtered_columns.append(col)
        column_indices.append(i)
    
    return filtered_columns, column_indices


def load_nprint_batch(file_paths, column_names, label, include_ip_port=True, max_rows=1024):
    """
    Load a batch of nprint files
    
    Args:
        file_paths (list): List of file paths to load
        column_names (list): Column names from the nprint files
        label (int): Label for the files (0 for real, 1 for synthetic)
        include_ip_port (bool): Whether to include IP/port columns
        max_rows (int): Maximum number of rows/packets to use from each file
        
    Returns:
        tuple: (X, y, apps, valid_indices)
            - X: Features matrix
            - y: Labels vector
            - apps: Application names
            - valid_indices: Indices of successfully processed files
    """
    num_files = len(file_paths)
    if num_files == 0:
        return np.array([]), np.array([]), np.array([]), []
    
    # Filter columns
    filtered_columns, column_indices = filter_columns(column_names, include_ip_port)
    
    # Each file has max_rows rows (packets), and each row has len(filtered_columns) features
    num_features_per_packet = len(filtered_columns)
    total_features = max_rows * num_features_per_packet
    
    # Pre-allocate arrays for better memory efficiency
    X = np.zeros((num_files, total_features), dtype=np.float32)
    y = np.empty(num_files, dtype=object)  # For application names
    apps = np.empty(num_files, dtype=object)
    valid_indices = []
    
    # Process each file
    for file_idx, file_path in enumerate(tqdm(file_paths, disable=len(file_paths)<10)):
        try:
            # Extract application name from filename
            filename = os.path.basename(file_path)
            application = filename.split('_')[0]  # Extract Amazon or Netflix
            apps[file_idx] = application
            
            # Initialize array for this file's data
            file_data = np.zeros(total_features, dtype=np.float32)
            
            # Read file line by line to save memory
            with open(file_path, 'r') as f:
                # Skip header
                next(f)
                
                # Read each line (packet)
                packet_idx = 0
                for line in f:
                    if packet_idx >= max_rows:
                        break
                        
                    values = line.strip().split(',')
                    #print("VALUES")
                    #print(values)
                    if len(values) != len(column_names):
                        print(f"Warning: Line {packet_idx+1} in {file_path} has incorrect number of fields")
                        continue
                    
                    # Extract only the values for the filtered columns
                    filtered_values = [values[i] for i in column_indices]
                    
                    # Convert values to floats and store in data array
                    feature_start = packet_idx * num_features_per_packet
                    feature_end = feature_start + num_features_per_packet
                    
                    file_data[feature_start:feature_end] = [float(x) if x != '-1' else -1.0 for x in filtered_values]
                    packet_idx += 1
                
                # If file has fewer than max_rows packets, fill with -1 (missing data)
                #if packet_idx < max_rows:
                #    print(f"Warning: {file_path} has only {packet_idx} packets instead of {max_rows} requested")
            
            # Add the file data to our dataset
            X[file_idx] = file_data
            y[file_idx] = application
            valid_indices.append(file_idx)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            #print("ERROR")
            # Don't add this file to valid indices
    
    # Return only valid data
    return X[valid_indices], y[valid_indices], apps[valid_indices], valid_indices


def generate_feature_names(column_names, max_rows, include_ip_port=True):
    """
    Generate feature names for filtered columns and specified number of rows
    
    Args:
        column_names (list): List of column names
        max_rows (int): Maximum number of rows to use
        include_ip_port (bool): Whether to include IP/port columns
        
    Returns:
        list: List of feature names
    """
    # First filter the columns
    filtered_columns, _ = filter_columns(column_names, include_ip_port)
    #for index, name in enumerate(filtered_columns):
    #    print(f"{index}: {name}")
    
    # Generate feature names for the filtered columns
    feature_names = []
    for i in range(max_rows):  # For each packet up to max_rows
        for col in filtered_columns:
            feature_names.append(f"{i}_{col}")
    for index, name in enumerate(feature_names):
        print(f"{index}: {name}")
    #print("\n\n\n\n FEATURE NAMES!!!!")
    #print(feature_names)
    #print("\n\n\n\n")
    
    return feature_names


def plot_feature_importance(feature_importance, feature_names, num_features=50, output_path=None):
    """
    Plot feature importance
    
    Args:
        feature_importance (array): Feature importance values
        feature_names (list): Feature names
        num_features (int): Number of top features to plot
        output_path (str): Path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Plot top features
    plt.figure(figsize=(15, 10))
    plt.title("Top Feature Importances - Random Forest", fontsize=16)
    plt.bar(range(min(num_features, len(indices))), 
            feature_importance[indices[:num_features]], 
            align="center")
    plt.xticks(range(min(num_features, len(indices))), 
               [feature_names[i] for i in indices[:num_features]], 
               rotation=90, 
               fontsize=8)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_feature_distributions(X_test, y_test, feature_names, indices, feature_importance, 
                               top_n=5, output_path=None):
    """
    Plot distributions of top features
    
    Args:
        X_test (array): Test data features
        y_test (array): Test data labels
        feature_names (list): Feature names
        indices (list): Sorted indices of features by importance
        feature_importance (array): Feature importance values
        top_n (int): Number of top features to plot
        output_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Get unique class names
    classes = np.unique(y_test)
    
    # Plot histograms for top features
    for i in range(min(top_n, len(indices))):
        feature_idx = indices[i]
        feature_name = feature_names[feature_idx]
        
        plt.subplot(top_n, 1, i+1)
        
        # Plot histogram for each class
        for class_name in classes:
            class_values = X_test[y_test == class_name, feature_idx]
            plt.hist(class_values, bins=30, alpha=0.5, label=class_name)
        
        plt.title(f"Distribution of {feature_name} (Importance: {feature_importance[feature_idx]:.6f})")
        plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"training_log_{timestamp}.txt")
    
    # Set up logging to file and console
    with open(log_file, 'w') as log:
        def log_print(message):
            print(message)
            log.write(f"{message}\n")
            log.flush()
            
        log_print(f"Starting training process at {timestamp}")
        log_print(f"Parameters: {vars(args)}")
        
        # Find all nprint files
        real_files = sorted(glob.glob(os.path.join(args.real_dir, "*.nprint")))
        synthetic_files = sorted(glob.glob(os.path.join(args.synthetic_dir, "*.nprint")))
        
        log_print(f"Found {len(real_files)} real nprint files")
        log_print(f"Found {len(synthetic_files)} synthetic nprint files")
        
        if not real_files:
            log_print("Error: No real nprint files found")
            return
            
        # Get column names from a sample file
        sample_file = real_files[0]
        column_names = get_column_names(sample_file)
        log_print(f"Found {len(column_names)} columns in nprint files")
        
        # Filter columns
        filtered_columns, column_indices = filter_columns(column_names, args.include_ip_port)
        log_print(f"After filtering, using {len(filtered_columns)} columns")
        log_print("Excluded columns related to IPv6, IPv4 options, and ICMP")
        if not args.include_ip_port:
            log_print("Also excluded IP address and port related columns")
        
        log_print(f"Using maximum of {args.max_rows} rows/packets per file")
        
        # Generate feature names only once
        feature_names = generate_feature_names(column_names, args.max_rows, args.include_ip_port)
        log_print(f"Generated {len(feature_names)} feature names")
        
        # Process real data in batches
        log_print("Processing real nprint files...")
        X_real_all = []
        y_real_all = []
        
        for i in range(0, len(real_files), args.batch_size):
            batch = real_files[i:i+args.batch_size]
            log_print(f"Processing batch {i//args.batch_size + 1}/{(len(real_files) + args.batch_size - 1)//args.batch_size} of real files")
            X_batch, y_batch, _, _ = load_nprint_batch(
                batch, column_names, 0, args.include_ip_port, args.max_rows
            )
            if len(X_batch) > 0:
                X_real_all.append(X_batch)
                y_real_all.append(y_batch)
        
        # Combine all real data batches
        X_real = np.vstack(X_real_all) if X_real_all else np.array([])
        y_real = np.concatenate(y_real_all) if y_real_all else np.array([])
        
        log_print(f"Total real dataset shape: {X_real.shape}")
        
        # Process synthetic data in batches
        X_synthetic = None
        y_synthetic = None
        
        if synthetic_files:
            log_print("Processing synthetic nprint files...")
            X_synthetic_all = []
            y_synthetic_all = []
            
            for i in range(0, len(synthetic_files), args.batch_size):
                batch = synthetic_files[i:i+args.batch_size]
                log_print(f"Processing batch {i//args.batch_size + 1}/{(len(synthetic_files) + args.batch_size - 1)//args.batch_size} of synthetic files")
                X_batch, y_batch, _, _ = load_nprint_batch(
                    batch, column_names, 1, args.include_ip_port, args.max_rows
                )
                if len(X_batch) > 0:
                    X_synthetic_all.append(X_batch)
                    y_synthetic_all.append(y_batch)
            
            # Combine all synthetic data batches
            X_synthetic = np.vstack(X_synthetic_all) if X_synthetic_all else np.array([])
            y_synthetic = np.concatenate(y_synthetic_all) if y_synthetic_all else np.array([])
            
            log_print(f"Total synthetic dataset shape: {X_synthetic.shape}")
        
        #----------------------------------------------------------------------
        # SCENARIO 1: Train on real data, test on real data
        #----------------------------------------------------------------------
        log_print("\n" + "="*80)
        log_print("SCENARIO 1: Train on real data, test on real data")
        log_print("="*80)
        
        # Split real data into train and test sets
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=args.test_size, random_state=args.random_state, stratify=y_real
        )
        
        log_print(f"Training set: {X_train_real.shape}")
        log_print(f"Testing set: {X_test_real.shape}")
        
        # Train Random Forest model
        log_print("Training Random Forest model on real data...")
        rf_model_real = RandomForestClassifier(
            n_estimators=args.rf_trees,
            random_state=args.random_state,
            n_jobs=-1,
            verbose=1 if args.verbose else 0
        )
        rf_model_real.fit(X_train_real, y_train_real)
        
        # Train Decision Tree model
        log_print("Training Decision Tree model on real data...")
        dt_model_real = DecisionTreeClassifier(random_state=args.random_state)
        dt_model_real.fit(X_train_real, y_train_real)
        
        # Evaluate models on real test data
        log_print("Evaluating models on real test data...")
        
        # Random Forest evaluation
        rf_pred_real = rf_model_real.predict(X_test_real)
        rf_accuracy_real = accuracy_score(y_test_real, rf_pred_real)
        rf_report_real = classification_report(y_test_real, rf_pred_real)
        rf_cm_real = confusion_matrix(y_test_real, rf_pred_real)
        
        log_print("\nReal Data - Random Forest Results:")
        log_print(f"Accuracy: {rf_accuracy_real:.4f}")
        log_print("\nClassification Report:")
        log_print(rf_report_real)
        log_print("\nConfusion Matrix:")
        log_print(str(rf_cm_real))
        
        # Decision Tree evaluation
        dt_pred_real = dt_model_real.predict(X_test_real)
        dt_accuracy_real = accuracy_score(y_test_real, dt_pred_real)
        dt_report_real = classification_report(y_test_real, dt_pred_real)
        dt_cm_real = confusion_matrix(y_test_real, dt_pred_real)
        
        log_print("\nReal Data - Decision Tree Results:")
        log_print(f"Accuracy: {dt_accuracy_real:.4f}")
        log_print("\nClassification Report:")
        log_print(dt_report_real)
        log_print("\nConfusion Matrix:")
        log_print(str(dt_cm_real))
        
        # Feature importance analysis for real data Random Forest
        log_print("\nFeature importance analysis for Real Data Random Forest model:")
        feature_importance_real = rf_model_real.feature_importances_
        
        # Sort features by importance
        indices_real = np.argsort(feature_importance_real)[::-1]
        
        # Print top 20 most important features
        log_print("\nTop 20 most important features (Real Data RF model):")
        for i in range(min(20, len(feature_names))):
            log_print(f"{feature_names[indices_real[i]]}: {feature_importance_real[indices_real[i]]:.6f}")
        
        # Plot feature importance for real data model
        plot_feature_importance(
            feature_importance_real,
            feature_names,
            num_features=50,
            output_path=os.path.join(args.output_dir, f"real_data_feature_importance_{timestamp}.png")
        )
        
        # Plot feature distributions for real data model
        plot_feature_distributions(
            X_test_real,
            y_test_real,
            feature_names,
            indices_real,
            feature_importance_real,
            top_n=args.visualize_top_n,
            output_path=os.path.join(args.output_dir, f"real_data_feature_distributions_{timestamp}.png")
        )
        
        # Save real data models
        joblib.dump(
            rf_model_real,
            os.path.join(args.output_dir, f"real_data_rf_model_{timestamp}.joblib")
        )
        joblib.dump(
            dt_model_real,
            os.path.join(args.output_dir, f"real_data_dt_model_{timestamp}.joblib")
        )
        
        # Save feature importance data
        pd.DataFrame({
            'feature_name': [feature_names[i] for i in indices_real],
            'importance': feature_importance_real[indices_real]
        }).to_csv(
            os.path.join(args.output_dir, f"real_data_feature_importance_{timestamp}.csv"),
            index=False
        )
        df_real_1 = pd.DataFrame(X_real)

        # Add the label column
        df_real_1['label'] = y_real

        # Write to CSV
        df_real_1.to_csv('real_dataset.csv', index=False)
        #----------------------------------------------------------------------
        # SCENARIO 2: Train on synthetic data, test on real data
        #----------------------------------------------------------------------
        if X_synthetic is not None and len(X_synthetic) > 0:
            log_print("\n" + "="*80)
            log_print("SCENARIO 2: Train on synthetic data, test on real data")
            log_print("="*80)
            
            # Use all synthetic data for training
            X_train_synthetic = X_synthetic
            y_train_synthetic = y_synthetic
            
            # Use all real data for testing
            X_test_synthetic = X_real
            y_test_synthetic = y_real
            
            df_syn_1 = pd.DataFrame(X_synthetic)

            # Add the label column
            df_syn_1['label'] = y_synthetic

            # Write to CSV
            df_syn_1.to_csv('synthetic_dataset.csv', index=False)
            
            df_real_1 = pd.DataFrame(X_real)

            # Add the label column
            df_real_1['label'] = y_real

            # Write to CSV
            df_real_1.to_csv('real_dataset.csv', index=False)
            
            
            log_print(f"Training set (synthetic): {X_train_synthetic.shape}")
            log_print(f"Testing set (real): {X_test_synthetic.shape}")
            
            # Train Random Forest model on synthetic data
            log_print("Training Random Forest model on synthetic data...")
            rf_model_synthetic = RandomForestClassifier(
                n_estimators=args.rf_trees,
                random_state=args.random_state,
                n_jobs=-1,
                verbose=1 if args.verbose else 0
            )
            rf_model_synthetic.fit(X_train_synthetic, y_train_synthetic)
            
            # Train Decision Tree model on synthetic data
            log_print("Training Decision Tree model on synthetic data...")
            dt_model_synthetic = DecisionTreeClassifier(random_state=args.random_state)
            dt_model_synthetic.fit(X_train_synthetic, y_train_synthetic)
            
            # Evaluate models on real data
            log_print("Evaluating synthetic-trained models on real data...")
            
            # Random Forest evaluation
            rf_pred_synthetic = rf_model_synthetic.predict(X_test_synthetic)
            rf_accuracy_synthetic = accuracy_score(y_test_synthetic, rf_pred_synthetic)
            rf_report_synthetic = classification_report(y_test_synthetic, rf_pred_synthetic)
            rf_cm_synthetic = confusion_matrix(y_test_synthetic, rf_pred_synthetic)
            
            log_print("\nSynthetic-trained Random Forest Results (tested on real data):")
            log_print(f"Accuracy: {rf_accuracy_synthetic:.4f}")
            log_print("\nClassification Report:")
            log_print(rf_report_synthetic)
            log_print("\nConfusion Matrix:")
            log_print(str(rf_cm_synthetic))
            
            # Decision Tree evaluation
            dt_pred_synthetic = dt_model_synthetic.predict(X_test_synthetic)
            dt_accuracy_synthetic = accuracy_score(y_test_synthetic, dt_pred_synthetic)
            dt_report_synthetic = classification_report(y_test_synthetic, dt_pred_synthetic)
            dt_cm_synthetic = confusion_matrix(y_test_synthetic, dt_pred_synthetic)
            
            log_print("\nSynthetic-trained Decision Tree Results (tested on real data):")
            log_print(f"Accuracy: {dt_accuracy_synthetic:.4f}")
            log_print("\nClassification Report:")
            log_print(dt_report_synthetic)
            log_print("\nConfusion Matrix:")
            log_print(str(dt_cm_synthetic))
            
            # Feature importance analysis for synthetic data Random Forest
            log_print("\nFeature importance analysis for Synthetic Data Random Forest model:")
            feature_importance_synthetic = rf_model_synthetic.feature_importances_
            
            # Sort features by importance
            indices_synthetic = np.argsort(feature_importance_synthetic)[::-1]
            
            # Print top 20 most important features
            log_print("\nTop 20 most important features (Synthetic Data RF model):")
            for i in range(min(20, len(feature_names))):
                log_print(f"{feature_names[indices_synthetic[i]]}: {feature_importance_synthetic[indices_synthetic[i]]:.6f}")
            
            # Plot feature importance for synthetic data model
            plot_feature_importance(
                feature_importance_synthetic,
                feature_names,
                num_features=50,
                output_path=os.path.join(args.output_dir, f"synthetic_data_feature_importance_{timestamp}.png")
            )
            
            # Plot feature distributions for synthetic data model
            plot_feature_distributions(
                X_test_synthetic,
                y_test_synthetic,
                feature_names,
                indices_synthetic,
                feature_importance_synthetic,
                top_n=args.visualize_top_n,
                output_path=os.path.join(args.output_dir, f"synthetic_data_feature_distributions_{timestamp}.png")
            )
            
            # Save synthetic data models
            joblib.dump(
                rf_model_synthetic,
                os.path.join(args.output_dir, f"synthetic_data_rf_model_{timestamp}.joblib")
            )
            joblib.dump(
                dt_model_synthetic,
                os.path.join(args.output_dir, f"synthetic_data_dt_model_{timestamp}.joblib")
            )
            
            # Save feature importance data
            pd.DataFrame({
                'feature_name': [feature_names[i] for i in indices_synthetic],
                'importance': feature_importance_synthetic[indices_synthetic]
            }).to_csv(
                os.path.join(args.output_dir, f"synthetic_data_feature_importance_{timestamp}.csv"),
                index=False
            )
            
            # Compare the two models
            log_print("\n" + "="*80)
            log_print("MODEL COMPARISON")
            log_print("="*80)
            log_print(f"Real data RF model accuracy on real data: {rf_accuracy_real:.4f}")
            log_print(f"Synthetic data RF model accuracy on real data: {rf_accuracy_synthetic:.4f}")
            log_print(f"Real data DT model accuracy on real data: {dt_accuracy_real:.4f}")
            log_print(f"Synthetic data DT model accuracy on real data: {dt_accuracy_synthetic:.4f}")
            
            # Compare feature importance between the two models
            log_print("\nComparing top 10 important features between models:")
            log_print("\nReal data model top 10 features:")
            for i in range(min(10, len(feature_names))):
                log_print(f"{i+1}. {feature_names[indices_real[i]]}: {feature_importance_real[indices_real[i]]:.6f}")
            
            log_print("\nSynthetic data model top 10 features:")
            for i in range(min(10, len(feature_names))):
                log_print(f"{i+1}. {feature_names[indices_synthetic[i]]}: {feature_importance_synthetic[indices_synthetic[i]]:.6f}")
            
            # Calculate overlap between top features
            top_k_values = [10, 20, 50, 100]
            for k in top_k_values:
                real_top_k = set([indices_real[i] for i in range(min(k, len(indices_real)))])
                synthetic_top_k = set([indices_synthetic[i] for i in range(min(k, len(indices_synthetic)))])
                overlap = real_top_k.intersection(synthetic_top_k)
                overlap_percentage = len(overlap) / k * 100 if k <= len(indices_real) and k <= len(indices_synthetic) else 0
                
                log_print(f"\nOverlap in top {k} features: {len(overlap)} features ({overlap_percentage:.2f}%)")
            
            # Plot comparison of feature importances
            plt.figure(figsize=(15, 10))
            plt.title("Top 20 Feature Importance Comparison", fontsize=16)
            
            # Get top 20 features from both models
            top_features = set()
            for i in range(min(20, len(indices_real))):
                top_features.add(indices_real[i])
            for i in range(min(20, len(indices_synthetic))):
                top_features.add(indices_synthetic[i])
            
            top_features = sorted(list(top_features))
            feature_labels = [feature_names[idx] for idx in top_features]
            
            # Get importances for these features
            real_importances = [feature_importance_real[idx] for idx in top_features]
            synthetic_importances = [feature_importance_synthetic[idx] for idx in top_features]
            
            # Bar positions
            x = np.arange(len(feature_labels))
            width = 0.35
            
            # Create bars
            plt.bar(x - width/2, real_importances, width, label='Real Data Model')
            plt.bar(x + width/2, synthetic_importances, width, label='Synthetic Data Model')
            
            plt.xticks(x, feature_labels, rotation=90, fontsize=8)
            plt.xlabel('Features', fontsize=14)
            plt.ylabel('Importance', fontsize=14)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            plt.savefig(
                os.path.join(args.output_dir, f"feature_importance_comparison_{timestamp}.png"),
                dpi=300
            )
            plt.close()
        else:
            log_print("\nSkipping Scenario 2 as no synthetic data was found or loaded.")
        
        log_print("\nTraining and evaluation complete!")
        log_print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
