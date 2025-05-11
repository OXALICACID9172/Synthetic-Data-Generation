"""
GAN with LSTM for Network Timestamp Generation (PyTorch)
-------------------------------------------------------
This code implements a Generative Adversarial Network (GAN) using LSTM
to generate synthetic network packet inter-arrival times from PCAP files.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scapy.all import rdpcap
import matplotlib.pyplot as plt
import dpkt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PCAPProcessor:
    def __init__(self, data_dir):
        """
        Initialize the PCAP processor.
        
        Args:
            data_dir: Directory containing PCAP files
        """
        self.data_dir = data_dir
        self.netflix_files = glob.glob(os.path.join(data_dir, "Netflix_*.pcap"))
        self.amazon_files = glob.glob(os.path.join(data_dir, "Amazon_*.pcap"))
        self.all_files = self.netflix_files + self.amazon_files
        self.max_values = []
        
    def extract_timestamps(self, pcap_file):
        """
        Extract inter-arrival times (IAT) from a PCAP file using dpkt.
        
        Args:
            pcap_file (str): Path to the PCAP file.
            
        Returns:
            np.ndarray: Inter-arrival times in microseconds (float32).
        """
        try:
            # Read the PCAP file with dpkt
            with open(pcap_file, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                timestamps = [ts for ts, _ in pcap]
                #print(timestamps)
            # Handle empty or single-packet capture
            if len(timestamps) <= 1:
                return np.array([], dtype=np.float32)
    
            # Compute inter-arrival times in microseconds
            base_time = timestamps[0]
            #iat = [(ts - base_time) * 1_000_000 for ts in timestamps]
            #return np.array(iat, dtype=np.float32)
            iat = [(timestamps[i] - timestamps[i - 1]) * 1_000_000 for i in range(1, len(timestamps))]
            return np.array(iat, dtype=np.float32)
        
        except Exception as e:
            print(f"Error processing {pcap_file}: {e}")
            return np.array([], dtype=np.float32)
    
    def process_all_files(self, sequence_length=64):
        """
        Process all PCAP files and prepare data for training.
        
        Args:
            sequence_length: Length of sequences for LSTM
            
        Returns:
            Normalized sequences, original max values
        """
        all_sequences = []
        max_values = []
        file_counter = 0
        for file in self.all_files:
            iat = self.extract_timestamps(file)
            #print("----IAT----")
            #print(iat)
            file_counter+=1
            print("Number of files processed:", file_counter, end='\r')
            if len(iat) < sequence_length:
                continue
                
            # Store the maximum value for this flow
            max_val = np.max(iat)
            if max_val > 0:  # Avoid division by zero
                max_values.append(max_val)
                
                # Normalize by dividing by maximum
                normalized_iat = iat / max_val
                
                # Create sequences
                for i in range(0, len(normalized_iat) - sequence_length + 1, sequence_length // 2):
                    seq = normalized_iat[i:i+sequence_length]
                    all_sequences.append(seq)
        
        self.max_values = np.array(max_values)
        return np.array(all_sequences), self.max_values
    
    def get_train_data(self, test_split=0.2, sequence_length=64):
        """
        Get training and testing data.
        
        Args:
            test_split: Fraction of data to use for testing
            sequence_length: Length of sequences for LSTM
            
        Returns:
            Training data, testing data, max values
        """
        sequences, max_values = self.process_all_files(sequence_length)
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences found in the PCAP files")
            
        # Shuffle the data
        indices = np.random.permutation(len(sequences))
        sequences = sequences[indices]
        
        # Split into training and testing sets
        split_idx = int(len(sequences) * (1 - test_split))
        train_data = sequences[:split_idx]
        test_data = sequences[split_idx:]
        
        return train_data, test_data, max_values


class TimestampDataset(Dataset):
    """Dataset for timestamp sequences."""
    
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]


class Generator(nn.Module):
    """LSTM-based generator for timestamp sequences."""
    
    def __init__(self, sequence_length, latent_dim, hidden_dim=128):
        super(Generator, self).__init__()
        
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        self.batch_norm2 = nn.BatchNorm1d(64)
        
        # Output layer
        self.output_layer = nn.Linear(64, 1)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        # First LSTM layer
        lstm1_out, _ = self.lstm1(z)
        # Apply batch normalization
        lstm1_out = self.batch_norm1(lstm1_out.transpose(1, 2)).transpose(1, 2)
        lstm1_out = self.leaky_relu(lstm1_out)
        
        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        # Apply batch normalization
        lstm2_out = self.batch_norm2(lstm2_out.transpose(1, 2)).transpose(1, 2)
        lstm2_out = self.leaky_relu(lstm2_out)
        
        # Output layer
        output = self.output_layer(lstm2_out)
        output = self.sigmoid(output)
        
        return output


class Discriminator(nn.Module):
    """LSTM-based discriminator for timestamp sequences."""
    
    def __init__(self, sequence_length):
        super(Discriminator, self).__init__()
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=128,  # 64*2 for bidirectional
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Dense layers
        self.fc1 = nn.Linear(64, 16)  # 32*2 for bidirectional
        self.fc2 = nn.Linear(16, 1)
        
        # Other layers
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # First LSTM layer
        lstm1_out, (h_n, _) = self.lstm1(x)
        lstm1_out = self.leaky_relu(lstm1_out)
        lstm1_out = self.dropout(lstm1_out)
        
        # Second LSTM layer
        _, (h_n, _) = self.lstm2(lstm1_out)
        
        # Concatenate the final hidden states from both directions
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        h_n = self.leaky_relu(h_n)
        h_n = self.dropout(h_n)
        
        # Dense layers
        fc1_out = self.fc1(h_n)
        fc1_out = self.leaky_relu(fc1_out)
        
        # Output layer
        validity = self.fc2(fc1_out)
        validity = self.sigmoid(validity)
        
        return validity


class MaxGenerator(nn.Module):
    """Generator for maximum values."""
    
    def __init__(self, latent_dim):
        super(MaxGenerator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensures positive values
        )
        
    def forward(self, z):
        return self.model(z)


class TimestampGAN:
    def __init__(self, sequence_length=64, latent_dim=100, lr=0.0002, beta1=0.5):
        """
        Initialize the GAN model.
        
        Args:
            sequence_length: Length of timestamp sequences
            latent_dim: Dimension of latent space
            lr: Learning rate
            beta1: Beta1 parameter for Adam optimizer
        """
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Initialize models
        self.generator = Generator(sequence_length, latent_dim).to(device)
        self.discriminator = Discriminator(sequence_length).to(device)
        self.max_generator = MaxGenerator(latent_dim).to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.max_g_optimizer = optim.Adam(self.max_generator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def train(self, train_data, epochs=2000, batch_size=32, sample_interval=100, save_interval=10, max_values=None):
        """
        Train the GAN model.
        
        Args:
            train_data: Training data (numpy array)
            epochs: Number of epochs
            batch_size: Batch size
            sample_interval: Interval for sampling and visualization
            save_interval: Interval for saving model checkpoints
            max_values: Array of maximum values for training max generator
        """
        # Prepare dataset and dataloader
        dataset = TimestampDataset(np.expand_dims(train_data, axis=-1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create directory for model checkpoints if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)
        
        # Arrays for recording history
        d_loss_history = []
        g_loss_history = []
        max_g_loss_history = []
        
        # Labels for real and fake samples
        real_label = 1
        fake_label = 0
        
        # Train the max value generator if we have max values
        if max_values is not None and len(max_values) > 0:
            max_values_tensor = torch.FloatTensor(max_values).to(device)
            max_g_epochs = epochs // 10 # Fewer epochs for max generator
            
            for epoch in range(max_g_epochs):
                # Sample random noise as generator input
                noise = torch.randn(batch_size, self.latent_dim).to(device)
                
                # Sample real max values
                idx = torch.randint(0, len(max_values), (batch_size,))
                real_max = max_values_tensor[idx].view(batch_size, 1)
                
                # Generate fake max values
                self.max_generator.zero_grad()
                gen_max = self.max_generator(noise)
                
                # Calculate loss and backpropagate
                max_g_loss = self.mse_loss(gen_max, real_max)
                max_g_loss.backward()
                self.max_g_optimizer.step()
                
                max_g_loss_history.append(max_g_loss.item())
                
                if epoch % sample_interval == 0:
                    print(f"[Max Generator] Epoch {epoch}, Loss: {max_g_loss.item()}")
                
                # Save the max generator model every save_interval epochs
                if epoch % save_interval == 0:
                    torch.save(
                        self.max_generator.state_dict(), 
                        f"checkpoints/max_generator_epoch_{epoch}.pth"
                    )
        
        # Main GAN training loop
        for epoch in range(epochs):
            for i, real_timestamps in enumerate(dataloader):
                batch_size = real_timestamps.size(0)
                
                # Move to device
                real_timestamps = real_timestamps.to(device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                self.d_optimizer.zero_grad()
                
                # Generate a batch of noise
                noise = torch.randn(batch_size, self.sequence_length, self.latent_dim).to(device)
                
                # Generate a batch of fake timestamps
                gen_timestamps = self.generator(noise)
                
                # Train with real timestamps
                real_validity = self.discriminator(real_timestamps)
                
                d_real_loss = self.adversarial_loss(
                    real_validity, 
                    torch.full((batch_size, 1), real_label, device=device).float()
                )
                
                # Train with fake timestamps
                fake_validity = self.discriminator(gen_timestamps.detach())
                d_fake_loss = self.adversarial_loss(
                    fake_validity, 
                    torch.full((batch_size, 1), fake_label, device=device).float()
                )
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                
                self.g_optimizer.zero_grad()
                
                # Generate new fake timestamps
                gen_timestamps = self.generator(noise)
                
                # Train generator to fool the discriminator
                validity = self.discriminator(gen_timestamps)
                g_loss = self.adversarial_loss(
                    validity, 
                    torch.full((batch_size, 1), real_label, device=device).float()
                )
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # Save losses for plotting
                d_loss_history.append(d_loss.item())
                g_loss_history.append(g_loss.item())
                
            # Print progress
            if epoch % sample_interval == 0:
                print(f"[Epoch {epoch}/{epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
                
                # Sample and visualize some generated timestamps
                self.sample_timestamps(epoch)
            
            # Save model checkpoints every save_interval epochs
            if epoch % save_interval == 0:
                print(f"Saving model checkpoint at epoch {epoch}")
                torch.save(
                    self.generator.state_dict(), 
                    f"checkpoints/generator_epoch_{epoch}.pth"
                )
                torch.save(
                    self.discriminator.state_dict(), 
                    f"checkpoints/discriminator_epoch_{epoch}.pth"
                )
                
        # Save the final trained models
        torch.save(self.generator.state_dict(), "generator_final.pth")
        torch.save(self.discriminator.state_dict(), "discriminator_final.pth")
        torch.save(self.max_generator.state_dict(), "max_generator_final.pth")
        
        # Plot the loss history
        self.plot_loss_history(d_loss_history, g_loss_history, max_g_loss_history)
        
        return d_loss_history, g_loss_history, max_g_loss_history
    
    def sample_timestamps(self, epoch, n_samples=4):
        """
        Sample and visualize some generated timestamps.
        
        Args:
            epoch: Current training epoch
            n_samples: Number of samples to generate
        """
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(n_samples, self.sequence_length, self.latent_dim).to(device)
            
            # Generate timestamps
            gen_timestamps = self.generator(noise).cpu().numpy()
            
            # Plot the generated timestamps
            fig, axs = plt.subplots(n_samples, 1, figsize=(10, 2*n_samples))
            for i in range(n_samples):
                axs[i].plot(gen_timestamps[i])
                axs[i].set_title(f"Generated timestamp sequence {i+1}")
                axs[i].set_xlabel("Packet index")
                axs[i].set_ylabel("Normalized IAT")
            
            plt.tight_layout()
            plt.savefig(f"generated_timestamps_epoch_{epoch}.png")
            plt.close()
    
    def plot_loss_history(self, d_loss_history, g_loss_history, max_g_loss_history):
        """
        Plot the training loss history.
        
        Args:
            d_loss_history: Discriminator loss history
            g_loss_history: Generator loss history
            max_g_loss_history: Max generator loss history
        """
        plt.figure(figsize=(10, 5))
        plt.plot(d_loss_history, label='Discriminator loss')
        plt.plot(g_loss_history, label='Generator loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("gan_loss_history.png")
        plt.close()
        
        if len(max_g_loss_history) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(max_g_loss_history, label='Max generator loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig("max_generator_loss_history.png")
            plt.close()
    
    def generate_timestamps(self, n_sequences=10, denormalize=True):
        """
        Generate timestamp sequences.
        
        Args:
            n_sequences: Number of sequences to generate
            denormalize: Whether to denormalize the timestamps
            
        Returns:
            Generated timestamp sequences
        """
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(n_sequences, self.sequence_length, self.latent_dim).to(device)
            
            # Generate timestamps
            gen_timestamps = self.generator(noise).cpu().numpy()
            
            # Generate max values for denormalization
            if denormalize:
                max_noise = torch.randn(n_sequences, self.latent_dim).to(device)
                max_values = self.max_generator(max_noise).cpu().numpy()
                
                # Denormalize the timestamps
                for i in range(n_sequences):
                    gen_timestamps[i] = gen_timestamps[i] * max_values[i]
            
            return gen_timestamps


def main():
    """Main function to run the training process."""
    
    # Parameters
    data_dir = "/home/cse/btech/cs1210575/scratch/time_stamp_gen/data/pcaps"  # Directory containing PCAP files
    sequence_length = 64
    latent_dim = 100
    batch_size = 32
    epochs = 2000
    sample_interval = 100
    save_interval = 10  # Save model checkpoints every 10 epochs
    
    # Process PCAP files
    processor = PCAPProcessor(data_dir)
    train_data, test_data, max_values = processor.get_train_data(
        test_split=0.2, 
        sequence_length=sequence_length
    )
    
    print(train_data)
    print(type(train_data))
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    print(f"Max values shape: {max_values.shape}")
    
    # Initialize and train GAN
    gan = TimestampGAN(
        sequence_length=sequence_length,
        latent_dim=latent_dim
    )
    
    # Train the GAN
    gan.train(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        sample_interval=sample_interval,
        save_interval=save_interval,
        max_values=max_values
    )
    
    # Generate some samples
    gen_timestamps = gan.generate_timestamps(n_sequences=10, denormalize=True)
    
    # Evaluate the generated samples
    # (You can add your own evaluation metrics here)
    
    # Plot some generated samples
    plt.figure(figsize=(12, 8))
    for i in range(min(5, len(gen_timestamps))):
        plt.subplot(5, 1, i+1)
        plt.plot(gen_timestamps[i])
        plt.title(f"Generated Timestamp Sequence {i+1}")
        plt.xlabel("Packet Index")
        plt.ylabel("Inter-Arrival Time")
    plt.tight_layout()
    plt.savefig("final_generated_timestamps.png")
    plt.show()
    
    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()