import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import math
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHANNEL_TYPE = 'AWGN'  

N_NORMAL = 7
K_NORMAL = 4
M_NORMAL = 2**K_NORMAL

EPOCHS_FINAL_TRAIN = 3000 

SNR_MIN_TEST = -20
SNR_MAX_TEST = 8
SNR_STEP_TEST = 1
N_TEST_BATCHES = 50      
TEST_BATCH_SIZE = 1000  


NUM_RUNS = 3  # Number of times to repeat the training/eval cycle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} for {CHANNEL_TYPE} channel simulation.")

# --- Helper Functions ---
def int_to_one_hot(ints, num_classes):
    one_hot = torch.zeros(ints.size(0), num_classes, device=ints.device)
    one_hot.scatter_(1, ints.unsqueeze(1), 1)
    return one_hot

def power_normalize(x):
    power = torch.mean(x**2, dim=1, keepdim=True)
    return x / torch.sqrt(power + 1e-8)

# --- Channel Simulation ---
def apply_fading(x, h):
    batch_size = x.shape[0]
    n_symbols = x.shape[1] // 2
    
    x_reshaped = x.reshape(batch_size, n_symbols, 2)
    x_complex = torch.view_as_complex(x_reshaped.contiguous()) 
    h_complex = torch.view_as_complex(h.contiguous())         
    y_complex = h_complex.unsqueeze(1) * x_complex
    y_real = torch.view_as_real(y_complex)
    return y_real.reshape(batch_size, -1)

def generate_fading_coefficient(batch_size, channel_type, device):
    if channel_type == 'Rayleigh':
        h_std = np.sqrt(0.5)
        h_real = torch.randn(batch_size, 1, device=device) * h_std
        h_imag = torch.randn(batch_size, 1, device=device) * h_std
    elif channel_type == 'Rician':
        k_factor =  10
        mean = np.sqrt(k_factor / (2 * (k_factor + 1)))
        std = np.sqrt(1 / (2 * (k_factor + 1)))
        h_real = torch.randn(batch_size, 1, device=device) * std + mean
        h_imag = torch.randn(batch_size, 1, device=device) * std + mean
    else:
        raise ValueError(f"Invalid channel type for fading: {channel_type}")
    
    return torch.cat((h_real, h_imag), dim=1)

def channel(x, ebn0_db, k_bits=4, channel_type='AWGN', h=None):
    """
    Applies channel effects using Eb/N0 as the noise parameter.
    """
    # --- 1. Calculate Rate (R) ---
    n_complex_symbols = x.shape[1] / 2.0
    rate = k_bits / n_complex_symbols 

    # --- 2. Convert Eb/N0 to SNR (Es/N0) ---
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    snr_linear = ebn0_linear * rate

    if isinstance(snr_linear, torch.Tensor) and snr_linear.ndim == 1:
        snr_linear = snr_linear.view(-1, 1)

    # --- 3. Apply Fading ---
    if channel_type in ['Rayleigh', 'Rician']:
        if h is None:
            h = generate_fading_coefficient(x.shape[0], channel_type, x.device)
        signal_after_fading = apply_fading(x, h)
    else:
        signal_after_fading = x
        h = None

    # --- 4. Add Noise ---
    signal_power = torch.mean(x**2, dim=1, keepdim=True)
    noise_variance = torch.zeros_like(signal_power)
    
    valid_mask = signal_power > 1e-10
    noise_variance[valid_mask] = signal_power[valid_mask] / snr_linear
    
    noise_std = torch.sqrt(noise_variance)
    noise = torch.randn_like(signal_after_fading) * noise_std
    
    y = signal_after_fading + noise
    
    return y, h


# --- Network Architectures ---

class ParameterEstimator(nn.Module):
    def __init__(self, n, n_units):
        super(ParameterEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * n, 2 * n), nn.ELU(),
            nn.Linear(2 * n, n_units), nn.Tanh(),
            nn.Linear(n_units, 2 * n), nn.Tanh(),
            nn.Linear(2*n, 2)
        )
    def forward(self, y):
        return self.net(y)

class UserEncoder(nn.Module):
    def __init__(self, m, n, n_channels):
        super(UserEncoder, self).__init__()
        self.n = n
        self.n_channels = n_channels
        self.dense_layers = nn.Sequential(
            nn.Linear(m, 2 * n), nn.ELU(),
            nn.Linear(2 * n, 2 * n), nn.ELU()
        )
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=2, stride=1), nn.Tanh(),
            nn.Conv1d(n_channels, n_channels, kernel_size=4, stride=2), nn.Tanh(),
            nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=1), nn.Tanh(),
            nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=1), nn.Tanh()
        )
        self.final_dense_in_features = self._get_conv_output_dim()
        self.final_dense = nn.Linear(self.final_dense_in_features, 2 * n)
    
    def _get_conv_output_dim(self):
        L_in = 2 * self.n 
        L_out = L_in - 2 + 1
        L_out = math.floor((L_out - 4) / 2 + 1)
        L_out = L_out - 2 + 1
        L_out = L_out - 2 + 1
        return self.n_channels * L_out

    def forward(self, s):
        x = self.dense_layers(s)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.final_dense(x)
        return power_normalize(x)

class UserDecoder(nn.Module):
    def __init__(self, m, n, channel_type, n_units_estimator, n_channels_decoder):
        super(UserDecoder, self).__init__()
        self.channel_type = channel_type
        self.n = n
        self.n_channels = n_channels_decoder
        self.estimator = ParameterEstimator(n, n_units_estimator)
        self.decoder_net = nn.Sequential(
            nn.Conv1d(1, n_channels_decoder, kernel_size=2, stride=1), nn.Tanh(),
            nn.Conv1d(n_channels_decoder, n_channels_decoder, kernel_size=4, stride=2), nn.Tanh(),
            nn.Conv1d(n_channels_decoder, n_channels_decoder, kernel_size=2, stride=1), nn.Tanh(),
            nn.Conv1d(n_channels_decoder, n_channels_decoder, kernel_size=2, stride=1), nn.Tanh()
        )
        self.dense_in_features = self._get_conv_output_dim()
        self.dense_layers = nn.Sequential(
            nn.Linear(self.dense_in_features, 2 * n), nn.Tanh(),
            nn.Linear(2 * n, m)
        )

    def _get_conv_output_dim(self):
        L_in = 2 * self.n 
        L_out = L_in - 2 + 1 
        L_out = math.floor((L_out - 4) / 2 + 1)
        L_out = L_out - 2 + 1
        L_out = L_out - 2 + 1
        return self.n_channels * L_out

    def forward(self, y):
        h_hat = self.estimator(y)
        y_processed = y
        
        if self.channel_type != 'AWGN':
            y_complex = torch.view_as_complex(y.reshape(y.shape[0], -1, 2))
            h_hat_complex = torch.view_as_complex(h_hat)
            y_equalized_complex = y_complex / (h_hat_complex.unsqueeze(1) + 1e-8)
            y_processed = torch.view_as_real(y_equalized_complex).reshape(y.shape[0], -1)

        y_processed = y_processed.unsqueeze(1)
        y_processed = self.decoder_net(y_processed)
        y_processed = y_processed.view(y_processed.size(0), -1)
        return self.dense_layers(y_processed)

# --- Training Function ---
def train_autoencoder_standalone(encoder, decoder, epochs, snr_db_min, snr_db_max, channel_type, lr, batch_size, run_idx):
    """ Trains the autoencoder with varying SNR. """
    print(f"\n[Run {run_idx+1}] Phase 1: Training (SNR: {snr_db_min} to {snr_db_max} dB)")
    encoder.to(DEVICE).train()
    decoder.to(DEVICE).train()
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6)
    
    start_time = time.time()
    for epoch in range(epochs):
        s_int_batch = torch.randint(0, M_NORMAL, (batch_size,), device=DEVICE)
        s_onehot_batch = int_to_one_hot(s_int_batch, M_NORMAL)
        
        current_snr = torch.rand(1).item() * (snr_db_max - snr_db_min) + snr_db_min
        
        optimizer.zero_grad()
        x = encoder(s_onehot_batch)
        y, _ = channel(x, current_snr, k_bits=K_NORMAL, channel_type=channel_type) 
        s_hat_logits = decoder(y)

        loss = criterion(s_hat_logits, s_int_batch)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if (epoch + 1) % 500 == 0:
            _, s_hat_int = torch.max(s_hat_logits, 1)
            bler = (s_hat_int != s_int_batch).float().mean().item()
            print(f"  Run {run_idx+1} | Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f} | BLER: {bler:.4f} | SNR: {current_snr:.1f} dB")
    
    end_time = time.time()
    print(f"  Run {run_idx+1} Training Complete ({end_time - start_time:.2f} s)")


# --- Evaluation Function ---
def evaluate_autoencoder(encoder, decoder, channel_type, snr_min_db, snr_max_db, snr_step, n_batches, batch_size, run_idx):
    """ Evaluates the trained autoencoder over a fixed SNR range. """
    print(f"\n[Run {run_idx+1}] Phase 2: Evaluation")
    encoder.to(DEVICE).eval()
    decoder.to(DEVICE).eval()

    snr_db_values = np.arange(snr_min_db, snr_max_db + snr_step, snr_step)
    bler_list = []
    
    total_test_blocks = n_batches * batch_size
    
    with torch.no_grad():
        for snr_db in snr_db_values:
            total_errors = 0
            for _ in range(n_batches):
                s_int_batch = torch.randint(0, M_NORMAL, (batch_size,), device=DEVICE)
                s_onehot_batch = int_to_one_hot(s_int_batch, M_NORMAL)
                
                x = encoder(s_onehot_batch)
                y, _ = channel(x, snr_db, k_bits=K_NORMAL, channel_type=CHANNEL_TYPE)
                s_hat_logits = decoder(y)
                
                _, s_hat_int = torch.max(s_hat_logits, 1)
                total_errors += (s_hat_int != s_int_batch).float().sum().item()

            bler = total_errors / total_test_blocks
            if bler == 0:
                bler = 1 / (total_test_blocks * 2) 
                
            bler_list.append(bler)
            # Optional: Print progress per SNR
            # print(f"  SNR: {snr_db:2d} dB | BLER: {bler:.6f}")
    
    print(f"  Run {run_idx+1} Evaluation finished.")
    return snr_db_values, bler_list

# --- Plotting Function ---
def plot_mean_bler_curve(snr_values, mean_bler, save_path="bler_plot_mean_final.png"):
    """ Plots the Mean BLER vs. SNR curve. """
    print(f"\n--- Phase 3: Plotting Mean Results ---")
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, mean_bler, marker='o', linestyle='-', linewidth=2, label=f'Mean BLER ({NUM_RUNS} runs)')
    
    plt.title(f'Mean BLER vs. SNR ({N_NORMAL},{K_NORMAL}) Autoencoder - {CHANNEL_TYPE}')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Block Error Rate (BLER)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.xlim(min(snr_values), max(snr_values))
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Best Hyperparameters from Optuna Study ---
    BEST_PARAMS = {
        'lr_ae': 0.006459275702578313,
        'batch_size': 128,
        'train_snr_db_min': 0,
        'train_snr_db_max': 4,
        'n_units_estimator': 32,
        'n_channels_ae': 7
    }
    
    print(f"--- Starting Multi-Run Experiment ({NUM_RUNS} Runs) ---")
    print(f"Parameters: {BEST_PARAMS}")
    
    all_run_blers = []
    snr_axis = None # Will store the SNR x-axis values

    # --- LOOP OVER RUNS ---
    for run_i in range(NUM_RUNS):
        print(f"\n{'='*20} STARTING RUN {run_i+1}/{NUM_RUNS} {'='*20}")
        
        # 1. Initialize networks (Fresh for each run!)
        user_encoder = UserEncoder(
            M_NORMAL, 
            N_NORMAL, 
            BEST_PARAMS['n_channels_ae']
        ).to(DEVICE)
        
        user_decoder = UserDecoder(
            M_NORMAL, 
            N_NORMAL, 
            CHANNEL_TYPE, 
            BEST_PARAMS['n_units_estimator'], 
            BEST_PARAMS['n_channels_ae']
        ).to(DEVICE)

        # 2. Run the training
        train_autoencoder_standalone(
            encoder=user_encoder,
            decoder=user_decoder,
            epochs=EPOCHS_FINAL_TRAIN, 
            snr_db_min=BEST_PARAMS['train_snr_db_min'],
            snr_db_max=BEST_PARAMS['train_snr_db_max'],
            channel_type=CHANNEL_TYPE,
            lr=BEST_PARAMS['lr_ae'],
            batch_size=BEST_PARAMS['batch_size'],
            run_idx=run_i
        )

        # 3. Run the evaluation
        snr_db, bler = evaluate_autoencoder(
            encoder=user_encoder,
            decoder=user_decoder,
            channel_type=CHANNEL_TYPE,
            snr_min_db=SNR_MIN_TEST,
            snr_max_db=SNR_MAX_TEST,
            snr_step=SNR_STEP_TEST,
            n_batches=N_TEST_BATCHES,
            batch_size=TEST_BATCH_SIZE,
            run_idx=run_i
        )
        
        # Store results
        all_run_blers.append(bler)
        if snr_axis is None:
            snr_axis = snr_db

    # --- 4. Calculate Mean and Print ---
    # Convert list of lists to numpy array for easy averaging
    all_run_blers = np.array(all_run_blers) # Shape: (NUM_RUNS, len(snr_axis))
    mean_bler = np.mean(all_run_blers, axis=0)
    
    print("\n" + "="*50)
    print(f"FINAL RESULTS: MEAN BLER OVER {NUM_RUNS} RUNS")
    print("="*50)
    print(f"{'SNR (dB)':<10} | {'Mean BLER':<15}")
    print("-" * 28)
    
    for i, snr_val in enumerate(snr_axis):
        print(f"{snr_val:<10.1f} | {mean_bler[i]:.8f}")
        
    print("="*50)
    
    # --- 5. Plot the Mean Results ---
    plot_mean_bler_curve(snr_axis, mean_bler)
    
    print("--- Experiment Complete ---")