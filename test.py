import os
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, Raw
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Extract general packet features
def extract_general_features(pkt):
    features = {}
    features['packet_length'] = len(pkt)  # Total packet size
    features['protocol'] = pkt[IP].proto if pkt.haslayer(IP) else None  # Protocol type (TCP=6, UDP=17)
    return features

# Extract TCP-specific features
def extract_tcp_features(pkt):
    features = {}
    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        features['tcp_src_port'] = tcp_layer.sport
        features['tcp_dst_port'] = tcp_layer.dport
        features['tcp_seq'] = tcp_layer.seq
        features['tcp_ack'] = tcp_layer.ack
        features['tcp_flags'] = tcp_layer.flags
        features['tcp_window_size'] = tcp_layer.window
        features['tcp_len'] = len(tcp_layer.payload) if tcp_layer.payload else 0
    return features

# Extract UDP-specific features
def extract_udp_features(pkt):
    features = {}
    if pkt.haslayer(UDP):
        udp_layer = pkt[UDP]
        features['udp_src_port'] = udp_layer.sport
        features['udp_dst_port'] = udp_layer.dport
        features['udp_len'] = len(udp_layer.payload) if udp_layer.payload else 0
    return features

# Process pcap files in a directory
def process_pcap_files(pcap_dir, is_real=True):
    all_features = []
    labels = []
    
    for label in ['Google', 'Facebook', 'Netflix']:
        label_dir = os.path.join(pcap_dir, label)
        if not os.path.exists(label_dir):
            print(label_dir)
            continue
        counter=0
        for file in os.listdir(label_dir):
            counter+=1
            if counter == 300:
                break
            print(counter)
            if file.endswith('.pcap'):
                pcap_file = os.path.join(label_dir, file)
                packets = rdpcap(pcap_file)  # Read packets using Scapy

                for pkt in packets:
                    # Check if the packet structure matches real or generated format
                    if is_real and not pkt.haslayer(IP):
                        continue  # Real packets should have IP
                    if not is_real and not pkt.haslayer(Raw):
                        continue  # Generated packets should have Raw

                    # Extract features
                    features = extract_general_features(pkt)
                    features.update(extract_tcp_features(pkt))
                    features.update(extract_udp_features(pkt))

                    all_features.append(features)
                    labels.append(label)

    return pd.DataFrame(all_features), labels

# Define your paths
real_data_dir = '/home/etbert/test_dir'
generated_data_dir = '/home/etbert/netDiffusion/NetDiffusion_Generator/final_data'

# Extract features
real_features, real_labels = process_pcap_files(real_data_dir, is_real=True)
generated_features, generated_labels = process_pcap_files(generated_data_dir, is_real=False)

# Ensure column consistency
generated_features = generated_features.reindex(columns=real_features.columns, fill_value=0)

# Case 1: Train and test on real data
X_train, X_test, y_train, y_test = train_test_split(real_features, real_labels, test_size=0.3, random_state=42)


# Case 2: Train on generated data and test on real data
X_train_gen, X_test_real, y_train_gen, y_test_real = generated_features, X_test, generated_labels, y_test

le = LabelEncoder()
all_flags = pd.concat([X_train['tcp_flags'], X_train_gen['tcp_flags']])
le.fit(all_flags)

X_train['tcp_flags'] = le.transform(X_train['tcp_flags'])
X_train_gen['tcp_flags'] = le.transform(X_train_gen['tcp_flags'])

print(X_train.dtypes)


# Fill NaNs with -1 or 'missing' for numerical or categorical respectively
imputer = SimpleImputer(strategy='constant', fill_value=-1)  # or 'missing' for strings
X_train = imputer.fit_transform(X_train)
X_train_gen = imputer.fit_transform(X_train_gen)


# Define models
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC()
}

# Function to evaluate models
def evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Classification Report for {model.__class__.__name__}:\n')
    print(classification_report(y_test, y_pred))

# Case 1: Train and test on real data
for name, model in models.items():
    print(f'\n{name} - Case 1: Train and Test on Real Data')
    evaluate_model(X_train, X_test, y_train, y_test, model)

# Case 2: Train on generated data and test on real data
for name, model in models.items():
    print(f'\n{name} - Case 2: Train on Generated Data and Test on Real Data')
    evaluate_model(X_train_gen, X_test, y_train_gen, y_test, model)
