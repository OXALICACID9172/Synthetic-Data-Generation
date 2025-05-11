"""
Simple Timestamp GAN Evaluation
-------------------------------
This script calculates three statistical distance metrics between real and generated timestamp data:
- Jensen-Shannon Divergence (JSD)
- Total Variation Distance (TD)
- Hellinger Distance (HD)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import os
import argparse
from sklearn.neighbors import KernelDensity

# Import your existing GAN code
# Make sure the paste.py file (with your GAN code) is in the same directory
from gan_gen import PCAPProcessor, TimestampGAN

def compute_distance_metrics(real_data, generated_data, n_bins=100, bandwidth=0.1):
    """
    Compute JSD, TD, and HD between real and generated timestamp data.
    
    Args:
        real_data (np.ndarray): Real timestamp data (flattened)
        generated_data (np.ndarray): Generated timestamp data (flattened)
        n_bins (int): Number of bins for histogram
        bandwidth (float): Bandwidth for KDE
        
    Returns:
        tuple: (jsd, td, hd) - Distance metrics
    """
    # Remove any NaN or infinite values
    real_data = real_data[np.isfinite(real_data)]
    generated_data = generated_data[np.isfinite(generated_data)]
    print(real_data)
    # Determine common range for consistent binning
    min_val = min(np.min(real_data), np.min(generated_data))
    max_val = max(np.max(real_data), np.max(generated_data))
    
    # Create histogram bins
    bins = np.linspace(min_val, max_val, n_bins)
    
    # Option 1: Using histograms (simpler but less accurate)
    real_hist, _ = np.histogram(real_data, bins=bins, density=True)
    gen_hist, _ = np.histogram(generated_data, bins=bins, density=True)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    real_hist = real_hist + epsilon
    gen_hist = gen_hist + epsilon
    
    # Normalize to ensure they sum to 1
    real_hist = real_hist / np.sum(real_hist)
    gen_hist = gen_hist / np.sum(gen_hist)
    
    # Calculate Jensen-Shannon Divergence
    jsd = jensenshannon(real_hist, gen_hist, base=2)
    
    # Calculate Total Variation Distance (TD)
    td = 0.5 * np.sum(np.abs(real_hist - gen_hist))
    
    # Calculate Hellinger Distance (HD)
    hd = np.sqrt(0.5 * np.sum((np.sqrt(real_hist) - np.sqrt(gen_hist)) ** 2))
    
    # Plot the histograms for visualization
    plt.figure(figsize=(12, 6))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.bar(bin_centers, real_hist, width=(max_val-min_val)/n_bins, alpha=0.5, label='Real Data')
    plt.bar(bin_centers, gen_hist, width=(max_val-min_val)/n_bins, alpha=0.5, label='Generated Data')
    plt.title(f'Distribution Comparison (JSD={jsd:.4f}, TD={td:.4f}, HD={hd:.4f})')
    plt.xlabel('Inter-Arrival Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_comparison_histogram.png')
    
    # Option 2: Using KDE for smoother distributions
    x_eval = np.linspace(min_val, max_val, n_bins).reshape(-1, 1)
    
    # Fit KDE for real data
    real_kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    real_kde.fit(real_data.reshape(-1, 1))
    real_kde_pdf = np.exp(real_kde.score_samples(x_eval))
    real_kde_pdf = real_kde_pdf / np.sum(real_kde_pdf)
    
    # Fit KDE for generated data
    gen_kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    gen_kde.fit(generated_data.reshape(-1, 1))
    gen_kde_pdf = np.exp(gen_kde.score_samples(x_eval))
    gen_kde_pdf = gen_kde_pdf / np.sum(gen_kde_pdf)
    
    # Calculate metrics using KDE
    jsd_kde = jensenshannon(real_kde_pdf, gen_kde_pdf, base=2)
    td_kde = 0.5 * np.sum(np.abs(real_kde_pdf - gen_kde_pdf))
    hd_kde = np.sqrt(0.5 * np.sum((np.sqrt(real_kde_pdf) - np.sqrt(gen_kde_pdf)) ** 2))
    
    # Plot the KDE distributions
    plt.figure(figsize=(12, 6))
    plt.plot(x_eval, real_kde_pdf, label='Real Data (KDE)', color='blue')
    plt.plot(x_eval, gen_kde_pdf, label='Generated Data (KDE)', color='red')
    plt.title(f'KDE Distribution Comparison (JSD={jsd_kde:.4f}, TD={td_kde:.4f}, HD={hd_kde:.4f})')
    plt.xlabel('Inter-Arrival Time')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_comparison_kde.png')
    
    # Return both histogram-based and KDE-based metrics
    metrics = {
        "histogram": {
            "jsd": jsd,
            "td": td,
            "hd": hd
        },
        "kde": {
            "jsd": jsd_kde,
            "td": td_kde,
            "hd": hd_kde
        }
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Calculate distance metrics between real and generated timestamps')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing PCAP files')
    parser.add_argument('--generator_path', type=str, default='generator_final.pth', help='Path to trained generator model')
    parser.add_argument('--max_generator_path', type=str, default='max_generator_final.pth', help='Path to trained max generator model')
    parser.add_argument('--sequence_length', type=int, default=64, help='Length of timestamp sequences')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent space')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--n_bins', type=int, default=100, help='Number of bins for histogram')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load real data
    print("Loading real data...")
    try:
        processor = PCAPProcessor(args.data_dir)
        _, test_data, max_values = processor.get_train_data(
            test_split=0.2,
            sequence_length=args.sequence_length
        )
        real_timestamps = test_data
    except Exception as e:
        print(f"Error loading real data: {e}")
        return
    
    # Initialize and load GAN model
    print("Loading GAN model...")
    try:
        gan = TimestampGAN(
            sequence_length=args.sequence_length,
            latent_dim=args.latent_dim
        )
        
        # Load trained models
        gan.generator.load_state_dict(torch.load(args.generator_path, map_location=device))
        gan.generator.eval()
        
        gan.max_generator.load_state_dict(torch.load(args.max_generator_path, map_location=device))
        gan.max_generator.eval()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Generate timestamps
    print(f"Generating {args.n_samples} timestamp sequences...")
    try:
        generated_timestamps = gan.generate_timestamps(n_sequences=args.n_samples, denormalize=True)
    except Exception as e:
        print(f"Error generating timestamps: {e}")
        return
    
    # Flatten for easier processing
    real_flat = real_timestamps.flatten()
    gen_flat = generated_timestamps.flatten()
    
    # Calculate metrics
    print("Calculating distance metrics...")
    metrics = compute_distance_metrics(real_flat, gen_flat, n_bins=args.n_bins)
    
    # Print results
    print("\n==== Distance Metrics Results ====")
    print("\nHistogram-based metrics:")
    print(f"Jensen-Shannon Divergence (JSD): {metrics['histogram']['jsd']:.6f}")
    print(f"Total Variation Distance (TD): {metrics['histogram']['td']:.6f}")
    print(f"Hellinger Distance (HD): {metrics['histogram']['hd']:.6f}")
    
    print("\nKDE-based metrics (smoother estimation):")
    print(f"Jensen-Shannon Divergence (JSD): {metrics['kde']['jsd']:.6f}")
    print(f"Total Variation Distance (TD): {metrics['kde']['td']:.6f}")
    print(f"Hellinger Distance (HD): {metrics['kde']['hd']:.6f}")
    
    print("\nPlots saved as 'distribution_comparison_histogram.png' and 'distribution_comparison_kde.png'")

if __name__ == "__main__":
    main()