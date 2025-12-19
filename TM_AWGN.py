import os
import os
# THESE LINES MUST BE FIRST, before tmu or anything else
os.environ['NUMBA_DEBUG'] = '0'
os.environ['NUMBA_DISABLE_DEBUG_LOGGING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from numba import njit
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.models.regression.vanilla_regressor import TMRegressor
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
import logging

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import torch

import numpy as np
import matplotlib.pyplot as plt

import time
 # Added for path handling
import joblib
from scipy.stats import norm


model_decoder = 'normal_decoder.joblib'
model_encoder_prefix = 'normal_encoder_'

@njit(debug=False)
def preprosses_TM_Decoder(y_np, n, num_bands=5):
    """
    Vectorized preprocessing of the received signal y for TM Classifier.
    Assumes y is in the range [-3, 3].

    y_np: (batch_size, 2*n) - received signal as a NumPy array
    n: original 'n' parameter
    num_bands: (int) - the number of bands for thermometer encoding
    """
    batch_size = y_np.shape[0]
    num_features = 2 * n

    # 1. Scale, round, and clamp
    # Map [-3, 3] to [0, num_bands]
    val_scaled = (y_np + 3.0) * (num_bands / 6.0)
    
    # Use np.empty_like for the output of np.round, which Numba prefers
    val_int_f = np.empty_like(val_scaled)
    np.round(val_scaled, 0, val_int_f)
    val_int = val_int_f.astype(np.int32)
    
    # Clamp value between 0 and num_bands - 1
    val = np.clip(val_int, 0, num_bands - 1)  # Shape: (batch_size, num_features)

    # 2. Create thermometer encoding
    k = np.arange(num_bands, dtype=np.int32)
    
    # Numba supports this broadcasting operation
    y_thermometer_3d = (k[np.newaxis, np.newaxis, :] <= val[..., np.newaxis])

    # 3. Reshape to final (batch_size, num_features * num_bands)
    y_thermometer = y_thermometer_3d.reshape(batch_size, num_features * num_bands).astype(np.uint32)
    
    return y_thermometer



@njit(debug=False)
def generate_perturbations_batch(y_batch_thermometer, n, num_bands):
    """
    Creates a batch of variations from a BATCH of thermometer-encoded instances
    by moving one "level" (one float feature) up or down at a time.
    
    Args:
        y_batch_thermometer (np.ndarray): Shape [B, 2*n*num_bands] (e.g., B, 80).
        n (int): The original 'n' parameter (e.g., 8).
        num_bands (int): The number of bands (e.g., 5).

    Returns:
        np.ndarray: A 2D uint32 array, shape [B * (1 + 2*2*n), 2*n*num_bands].
                    (e.g., B * 33, 80)
    """
    batch_size = y_batch_thermometer.shape[0]
    num_float_features = 2 * n  # e.g., 16
    
    # 1. Reshape to (B, N_float_features, N_bands)
    y_reshaped = y_batch_thermometer.reshape((batch_size, num_float_features, num_bands))
    
    # 2. Create the "mega-batch" by repeating
    num_perturbations_per_instance = 1 + 2 * num_float_features # e.g., 1 + 32 = 33
    
    # Shape: [B * 33, 16, 5]
    # y_mega_batch = np.repeat(y_reshaped, num_perturbations_per_instance, axis=0) # <-- Numba does not support axis
    
    # 3. Find the count of ones for each feature. Shape: (B, 16)
    # E.g., [1, 1, 1, 0, 0] -> num_ones = 3
    num_ones = np.sum(y_reshaped, axis=2).astype(np.int32) 
    
    # 4. Find the indices to flip
    # "Up" flip index: This is just num_ones. 
    # For [1, 1, 1, 0, 0], num_ones = 3. We flip index 3.
    # Clip to be a valid index (0 to num_bands-1)
    up_flip_idx = np.clip(num_ones, 0, num_bands - 1) # Shape: (B, 16)
    
    # "Down" flip index: This is num_ones - 1.
    # For [1, 1, 1, 0, 0], num_ones = 3. We flip index 2.
    down_flip_idx = np.clip(num_ones - 1, 0, num_bands - 1) # Shape: (B, 16)
    
    # 5. Tile indices to (B * 33, 16)
    # up_flip_idx_tiled = np.tile(up_flip_idx, (num_perturbations_per_instance, 1)) # <-- Numba does not support tile
    # down_flip_idx_tiled = np.tile(down_flip_idx, (num_perturbations_per_instance, 1)) # <-- Numba does not support tile

    # --- Numba-friendly replacement for np.repeat(axis=0) and np.tile ---
    
    # 2. Create the "mega-batch"
    # Shape: [B * 33, 16, 5]
    y_mega_batch = np.empty(
        (batch_size * num_perturbations_per_instance, num_float_features, num_bands), 
        dtype=y_reshaped.dtype
    )
    
    # 5. Create tiled indices
    up_flip_idx_tiled = np.empty(
        (batch_size * num_perturbations_per_instance, num_float_features), 
        dtype=up_flip_idx.dtype
    )
    down_flip_idx_tiled = np.empty(
        (batch_size * num_perturbations_per_instance, num_float_features), 
        dtype=down_flip_idx.dtype
    )

    # Loop to manually perform repeat and tile
    for i in range(batch_size):
        start_row = i * num_perturbations_per_instance
        end_row = (i + 1) * num_perturbations_per_instance
        
        # Manually repeat y_reshaped[i]
        for j in range(start_row, end_row):
            y_mega_batch[j] = y_reshaped[i]
            
        # Manually tile up_flip_idx[i] and down_flip_idx[i]
        for j in range(start_row, end_row):
            up_flip_idx_tiled[j] = up_flip_idx[i]
            down_flip_idx_tiled[j] = down_flip_idx[i]
            
    # --- End of replacement ---
    
    # Get batch indices for all rows in the mega-batch
    # batch_indices = np.arange(y_mega_batch.shape[0]) # Not strictly needed in loop
    
    # 6. Loop over the 16 FLOAT features to apply perturbations
    for i in range(num_float_features):
        # 1. "Up" Perturbations (Level 0 -> 1)
        # Get the row indices for this feature's "up" move for ALL instances
        # This is row 1, 34, 67, ... for i=0
        # This is row 3, 36, 69, ... for i=1
        up_rows = np.arange(2 * i + 1, batch_size * num_perturbations_per_instance, num_perturbations_per_instance)
        
        # Get the band index to flip for each of these rows
        band_idx_to_flip = up_flip_idx_tiled[up_rows, i]
        
        # Apply the flip (e.g., [1,1,1,0,0] -> [1,1,1,1,0])
        # This is safe: if already [1,1,1,1,1], up_flip_idx=4, and it just sets y[4]=1, no change.
        # y_mega_batch[up_rows, i, band_idx_to_flip] = 1 # <-- Numba does not support advanced indexing
        
        # Numba-compatible loop
        for k in range(up_rows.shape[0]):
            row_idx = up_rows[k]
            band_idx = band_idx_to_flip[k]
            y_mega_batch[row_idx, i, band_idx] = 1

        # 2. "Down" Perturbations (Level 1 -> 0)
        # Get row indices for this feature's "down" move
        # This is row 2, 35, 68, ... for i=0
        # This is row 4, 37, 70, ... for i=1
        down_rows = np.arange(2 * i + 2, batch_size * num_perturbations_per_instance, num_perturbations_per_instance)
        
        # Get the band index to flip
        band_idx_to_flip = down_flip_idx_tiled[down_rows, i]
        
        # Apply the flip (e.g., [1,1,1,0,0] -> [1,1,0,0,0])
        # This is safe: if already [0,0,0,0,0], down_flip_idx=0, and it just sets y[0]=0, no change.
        # y_mega_batch[down_rows, i, band_idx_to_flip] = 0 # <-- Numba does not support advanced indexing
        
        # Numba-compatible loop
        for k in range(down_rows.shape[0]):
            row_idx = down_rows[k]
            band_idx = band_idx_to_flip[k]
            y_mega_batch[row_idx, i, band_idx] = 0

    # 7. Reshape back to (B * 33, 80)
    return y_mega_batch.reshape((batch_size * num_perturbations_per_instance, -1))

@njit(debug=False)
def calculate_tm_scores_batch(votesums_mega_batch, s_int_batch, num_classes):
    """
    Calculates the scores for all instances in the batch from the 
    raw vote sums from the TM.

    Args:
        votesums_mega_batch (np.ndarray): Shape [B * (1 + 2*N), M]
        s_int_batch (np.ndarray): Shape [B]
        num_classes (int): M

    Returns:
        np.ndarray: Shape [B, 1 + 2*N]
    """
    batch_size = s_int_batch.shape[0]
    num_features = (votesums_mega_batch.shape[0] // batch_size - 1) // 2
    
    # 1. Get the correct class label for each row in the mega-batch
    # Shape: [B * (1 + 2*N)]
    correct_class_repeated = np.repeat(s_int_batch, 1 + 2 * num_features)

    # 2. Get the votes for the correct class for all rows
    # Shape: [B * (1 + 2*N)]
    # all_rows_idx = np.arange(votesums_mega_batch.shape[0]) # <-- Not needed for loop
    # correct_class_votes = votesums_mega_batch[all_rows_idx, correct_class_repeated] # <-- Numba doesn't support this advanced indexing
    
    # --- Numba-compatible replacement for advanced indexing ---
    num_rows = votesums_mega_batch.shape[0]
    # Ensure correct dtype for the new array, matching the input votes.
    correct_class_votes = np.empty(num_rows, dtype=votesums_mega_batch.dtype)
    
    for i in range(num_rows):
        correct_class_idx = correct_class_repeated[i]
        correct_class_votes[i] = votesums_mega_batch[i, correct_class_idx]
    # --- End of replacement ---

    # 3. Get the total votes (sum of all other classes)
    # Shape: [B * (1 + 2*N)]
    total_votes = np.sum(votesums_mega_batch, axis=1)
    
    # 4. Calculate score: minimize (other_votes) - (correct_votes)
    # This is equivalent to minimizing (total_votes - correct_votes) - correct_votes
    # which is (total_votes - 2 * correct_class_votes)
    # The original code had:
    # score = -correct_class_votes + sum(class_votes) - correct_class_votes
    # scoures.append(-score)
    # scores_min = np.array(scoures)
    # This means score = sum(class_votes) - 2 * correct_class_votes
    # And scores_min = -(sum(class_votes) - 2 * correct_class_votes)
    # This is 2 * correct_class_votes - total_votes
    # Since the original function used 'scores_min', it seems it wanted to
    # MINIMIZE the value. A *lower* score is *better*.
    # Let's re-read create_best_variation: "LOWER score is BETTER"
    # So, score = total_votes - 2 * correct_class_votes
    scores_min_flat = total_votes - 2 * correct_class_votes
    
    # 5. Reshape to [B, 1 + 2*N]
    return scores_min_flat.reshape(batch_size, 1 + 2 * num_features)

@njit(debug=False)
def create_best_variation_batch(y_original_batch, scores_batch, n, num_bands):
    """
    Creates new 'y' thermometer vectors by combining the best move for each
    FLOAT FEATURE (level up, level down, or stay).

    Args:
        y_original_batch (np.ndarray): Shape [B, 2*n*num_bands] (e.g., B, 80)
        scores_batch (np.ndarray): Shape [B, 1 + 2*2*n] (e.g., B, 33)
        n (int): e.g., 8
        num_bands (int): e.g., 5

    Returns:
        np.ndarray: The new, optimized 2D 'y' vectors (shape [B, 2*n*num_bands]).
    """
    batch_size = y_original_batch.shape[0]
    num_float_features = 2 * n
    
    # Reshape original to (B, 16, 5)
    y_original_reshaped = y_original_batch.reshape((batch_size, num_float_features, num_bands))
    y_new_batch = y_original_reshaped.copy()
    
    # Get baseline scores. Shape: (B, 1)
    baseline_scores = scores_batch[:, 0:1]
    
    # Get num_ones to find valid indices. Shape: (B, 16)
    num_ones = np.sum(y_original_reshaped, axis=2).astype(np.int32)
    
    baseline_scores_flat = baseline_scores[:, 0] # Use flat version for comparison

    for i in range(num_float_features):
        # Get scores for this feature's moves
        # Shape: (B,)
        scores_up = scores_batch[:, 2 * i + 1]
        scores_down = scores_batch[:, 2 * i + 2]
        
        # Find best score for this feature. Shape: (B,)
        # Use baseline_scores[:, 0] instead of .squeeze()
        # best_local_score = np.minimum(np.minimum(baseline_scores[:, 0], scores_up), scores_down)
        
        # Numba-friendly minimum calculation
        best_local_score_0 = np.minimum(baseline_scores_flat, scores_up)
        best_local_score = np.minimum(best_local_score_0, scores_down)
        
        # Get the indices to flip (if we need to)
        # Shape: (B,)
        up_flip_idx = np.clip(num_ones[:, i], 0, num_bands - 1)
        down_flip_idx = np.clip(num_ones[:, i] - 1, 0, num_bands - 1)
        
        # Get batch indices
        # batch_idx = np.arange(batch_size) # <-- Not needed in loop
        
        # Condition 1: "Up" (level+) is the best move
        # AND it's a valid move (not already at max)
        # Use baseline_scores[:, 0] instead of .squeeze()
        # cond_up = (best_local_score == scores_up) & \ # <-- Vectorized boolean ops
        #           (scores_up < baseline_scores[:, 0]) & \
        #           (num_ones[:, i] < num_bands)
                  
        # Condition 2: "Down" (level-) is the best move
        # AND it's a valid move (not already at min)
        # Use baseline_scores[:, 0] instead of .squeeze()
        # cond_down = (best_local_score == scores_down) & \ # <-- Vectorized boolean ops
        #             (scores_down < baseline_scores[:, 0]) & \
        #             (num_ones[:, i] > 0)
        
        # Apply the best moves for this feature *across the whole batch*
        # if np.any(cond_up): # <-- Numba does not support advanced indexing
        #     rows_to_mod = batch_idx[cond_up]
        #     band_idx_to_mod = up_flip_idx[cond_up]
        #     y_new_batch[rows_to_mod, i, band_idx_to_mod] = 1
            
        # if np.any(cond_down): # <-- Numba does not support advanced indexing
        #     rows_to_mod = batch_idx[cond_down]
        #     band_idx_to_mod = down_flip_idx[cond_down]
        #     y_new_batch[rows_to_mod, i, band_idx_to_mod] = 0
            
        # --- Numba-compatible replacement for advanced indexing ---
        # Loop over each item in the batch
        for b_idx in range(batch_size):
            # Check up condition
            cond_up = (best_local_score[b_idx] == scores_up[b_idx]) & \
                      (scores_up[b_idx] < baseline_scores_flat[b_idx]) & \
                      (num_ones[b_idx, i] < num_bands)
            
            # Check down condition
            cond_down = (best_local_score[b_idx] == scores_down[b_idx]) & \
                        (scores_down[b_idx] < baseline_scores_flat[b_idx]) & \
                        (num_ones[b_idx, i] > 0)
            
            if cond_up:
                band_idx = up_flip_idx[b_idx]
                y_new_batch[b_idx, i, band_idx] = 1
            
            if cond_down:
                band_idx = down_flip_idx[b_idx]
                y_new_batch[b_idx, i, band_idx] = 0
        # --- End of replacement ---

    return y_new_batch.reshape((batch_size, -1))

@njit(debug=False)
def reverse_preprosses_TM_Decoder_Selector(y_thermometer, n, num_bands=5, min_val=-4.0, max_val=4.0, mode='middle'):
    """
    Reverses thermometer encoding with selectable reconstruction strategies.

    Args:
        y_thermometer: Input binary array.
        n: Feature parameter.
        num_bands: Number of bands.
        min_val, max_val: Range of the signal.
        mode (str): Strategy for picking the point:
            - 'lower': Pick the floor of the band (Original behavior).
            - 'upper': Pick the ceiling of the band.
            - 'middle': Pick the exact center of the band.
            - 'uniform': Random point uniformly distributed in the band.
            - 'gaussian': Random point centered in the band with Gaussian noise.

    Returns:
        np.ndarray: Reconstructed float values.
    """
    batch_size = y_thermometer.shape[0]
    num_features = 2 * n
    
    # 1. Calculate geometry
    span = max_val - min_val
    band_width = span / num_bands

    # 2. Decode thermometer counts
    # Reshape to (Batch, Features, Bands)
    reshaped_array = y_thermometer.reshape((batch_size, num_features, num_bands))
    
    # Count ones to find the band index (0 to num_bands-1)
    num_ones = np.sum(reshaped_array, axis=2)
    val_array = num_ones - 1

    # 3. Calculate Base (Lower Bound)
    # This gives the "start" of the specific band
    y_base = (val_array.astype(np.float64) * band_width) + min_val

    # 4. Apply Mode Strategy
    y_final = np.zeros_like(y_base)

    if mode == 'lower':
        # Just the base value
        y_final = y_base

    elif mode == 'upper':
        # Base + full width
        y_final = y_base + band_width

    elif mode == 'middle':
        # Base + half width
        y_final = y_base + (0.5 * band_width)

    elif mode == 'uniform':
        # Add random noise between 0 and band_width
        # np.random.random() returns [0.0, 1.0)
        noise = np.random.random(y_base.shape)
        y_final = y_base + (noise * band_width)

    elif mode == 'gaussian':
        # Center the gaussian in the middle of the band
        center_offset = 0.5 * band_width
        
        # Standard deviation: Set so that +/- 3 sigma covers the band width
        # This keeps ~99% of values inside the band naturally
        sigma = band_width / 6.0 
        
        # Generate gaussian noise
        noise = np.random.normal(0.0, 1.0, size=y_base.shape) * sigma
        
        y_final = y_base + center_offset + noise

    else:
        # Fallback to middle if typo
        y_final = y_base + (0.5 * band_width)

    # 5. Global Clip
    # Essential for 'upper' (which hits the max) and 'gaussian' (tails might go out)
    y_final = np.clip(y_final, min_val, max_val)

    return y_final


# --- Configuration ---
CHANNEL_TYPE = 'AWGN'  # Options: 'AWGN', 'Rayleigh', 'Rician'

# Autoencoder (Normal Communication) Parameters
N_NORMAL = 2
K_NORMAL = 4
M_NORMAL = 2**K_NORMAL



# Training Parameters
BATCH_SIZE = 5000
LEARNING_RATE_AE = 0.24320239349059608
EPOCHS_AE = 40
num_bands = 40
STANDARDIZE_EPOCHS = 40


# Define evaluation parameters
SNR_MIN = -4
SNR_MAX = 8
N_BATCHES = 10
B_SIZE = 1000

if CHANNEL_TYPE == 'AWGN':
    AE_TRAIN_SNR_DB = 6  # This is not in the new params, but kept from original
    TRAIN_SNR_DB_MIN = -2
    TRAIN_SNR_DB_MAX = 8
else:  # Rayleigh or Rician
    AE_TRAIN_SNR_DB = 8
    TRAIN_SNR_DB_MIN = 15
    TRAIN_SNR_DB_MAX = 30


DEVICE = torch.device("cuda" if 0 else "cpu")
print(f"Using device: {DEVICE} for {CHANNEL_TYPE} channel simulation.")


# --- Helper Functions ---
def int_to_one_hot(ints, num_classes):
    """Converts a tensor of integers to a one-hot tensor."""
    one_hot = torch.zeros(ints.size(0), num_classes, device=ints.device)
    one_hot.scatter_(1, ints.unsqueeze(1), 1)
    return one_hot

def power_normalize(x):
    """Normalizes the power of a batch of signals to 1."""
    power = torch.mean(x**2, dim=1, keepdim=True)
    return x / torch.sqrt(power + 1e-8), power

# --- Channel Simulation ---
def apply_fading(x, h):
    """
    Apply complex multiplicative fading.
    x: (batch_size, 2*n) - signal with n symbols, each symbol is 2 real values (I,Q)
    h: (batch_size, 2) - ONE complex fading coefficient per batch item
    Returns: (batch_size, 2*n) - faded signal
    """
    batch_size = x.shape[0]
    n_symbols = x.shape[1] // 2

    x_reshaped = x.reshape(batch_size, n_symbols, 2)
    x_complex = torch.view_as_complex(x_reshaped.contiguous())
    h_complex = torch.view_as_complex(h.contiguous())
    
    y_complex = h_complex.unsqueeze(1) * x_complex
    y_real = torch.view_as_real(y_complex)
    return y_real.reshape(batch_size, -1)

def generate_fading_coefficient(batch_size, channel_type, device):
    """
    Generate a fading coefficient for Rayleigh or Rician channels.
    Returns: (batch_size, 2) tensor representing complex fading coefficient
    """
    if channel_type == 'Rayleigh':
        h_std = np.sqrt(0.5)
        h_real = torch.randn(batch_size, 1, device=device) * h_std
        h_imag = torch.randn(batch_size, 1, device=device) * h_std
    elif channel_type == 'Rician':
        k_factor = 10
        mean = np.sqrt(k_factor / (2 * (k_factor + 1)))
        std = np.sqrt(1 / (2 * (k_factor + 1)))
        h_real = torch.randn(batch_size, 1, device=device) * std + mean
        h_imag = torch.randn(batch_size, 1, device=device) * std + mean
    else:
        raise ValueError(f"Invalid channel type for fading: {channel_type}")

    return torch.cat((h_real, h_imag), dim=1)


def channel(x, ebn0_db, k_bits = 4, channel_type='AWGN', h=None):
    """
    Applies channel effects using Eb/N0 as the noise parameter.
    
    Args:
        x: Input tensor of shape (Batch_Size, 2*N_symbols)
        ebn0_db: Energy per Bit to Noise Power Spectral Density ratio (in dB)
        k_bits: Number of information bits per block (e.g., K_NORMAL)
        channel_type: 'AWGN', 'Rayleigh', or 'Rician'
        h: Fading coefficients (optional)
    """
    
    # --- 1. Calculate Rate (R) ---
    # We assume x contains complex symbols split into 2 real dimensions.
    # Shape of x is [Batch, 2*N], so number of complex symbols is x.shape[1] / 2
    n_complex_symbols = x.shape[1] / 2.0
    
    # Rate = Bits / Complex Symbol
    rate = k_bits / n_complex_symbols 

    # --- 2. Convert Eb/N0 to SNR (Es/N0) ---
    # Formula: SNR (linear) = (Eb/N0_linear) * Rate
    # In dB:   SNR_dB = Eb/N0_dB + 10*log10(Rate)
    
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    snr_linear = ebn0_linear * rate

    # Handle if ebn0_db was a tensor (per-sample noise)
    if isinstance(snr_linear, torch.Tensor) and snr_linear.ndim == 1:
        snr_linear = snr_linear.view(-1, 1)

    # --- 3. Apply Fading (Same as before) ---
    if channel_type in ['Rayleigh', 'Rician']:
        if h is None:
            h = generate_fading_coefficient(x.shape[0], channel_type, x.device)
        signal_after_fading = apply_fading(x, h)
    else:
        signal_after_fading = x
        h = None

    # --- 4. Add Noise based on calculated SNR ---
    signal_power = torch.mean(x**2, dim=1, keepdim=True)
    noise_variance = torch.zeros_like(signal_power)
    
    valid_mask = signal_power > 1e-10
    
    # Use the calculated snr_linear here
    noise_variance[valid_mask] = signal_power[valid_mask] / snr_linear
    
    noise_std = torch.sqrt(noise_variance)
    noise = torch.randn_like(signal_after_fading) * noise_std
    
    y = signal_after_fading + noise
    
    return y, h

# --- Training Functions ---
def train_autoencoder(tm_decoder, tm_encoders, epochs, snr_db, channel_type):
    """
    Trains the DNN autoencoder (Encoder + Decoder) AND the TM Decoder
    simultaneously on the same channel data.
    
    Refactored to be efficient by vectorizing the TM feedback loop.
    
    NOTE: This implementation follows the original code's logic, which *only*
    trains the ENCODER using the TM feedback loss. The DNN DECODER is
    *not* trained.
    """
    print(f"\n--- Phase 1: Pre-training Autoencoder and TM for {channel_type} ---")

    # Number of bands for thermometer encoding
    global LEARNING_RATE_AE
    # This optimizer *only* trains the encoder, per the original code's logic

    s_int_batch = torch.randint(0, M_NORMAL, (BATCH_SIZE,), device=DEVICE)
    s_onehot_batch = int_to_one_hot(s_int_batch, M_NORMAL)
    s_onehot_batch = s_onehot_batch.cpu().numpy()
    yyy = np.array([-3.0]*8 + [3.0]*8)


    print(yyy)
    for tm_encoder in tm_encoders:
        tm_encoder.init_after(X= s_onehot_batch, Y= yyy)
        tm_encoder.init_clause_bank(X= s_onehot_batch, Y= np.zeros((16,16)))
        tm_encoder.init_weight_bank(X= s_onehot_batch, Y= np.zeros((16,16)), )
    for epoch in range(epochs+1):
        s_int_batch = torch.randint(0, M_NORMAL, (BATCH_SIZE,), device=DEVICE)
        s_onehot_batch = int_to_one_hot(s_int_batch, M_NORMAL)

        current_snr = torch.rand(1).item() * (TRAIN_SNR_DB_MAX - TRAIN_SNR_DB_MIN) + TRAIN_SNR_DB_MIN 
        
        

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # --- Forward Pass ---
        x = np.zeros((BATCH_SIZE, 2*N_NORMAL))
        for i in range(2*N_NORMAL):
            x[:,i] = tm_encoders[i].predict(s_onehot_batch.cpu().numpy())

        

        x, power = power_normalize(torch.tensor(x))
      

        # --- fading 
        if channel_type in ['Rayleigh', 'Rician']:
            h = generate_fading_coefficient(BATCH_SIZE, channel_type, DEVICE)
            x = apply_fading(x, h)

            

        else:
            h = None




    
        x = x.cpu().numpy()
        signal_power = np.mean(x**2)
        rate = K_NORMAL / N_NORMAL

        snr_linear = 10**(current_snr / 10.0) * rate
        
        if signal_power.item() < 1e-10:
            noise_variance = 1.0 / snr_linear
        else:
            noise_variance = signal_power / snr_linear
            
        noise = torch.randn_like(torch.tensor(x)).numpy() * np.sqrt(noise_variance)
        y = x + noise
     
        # --- TM Training ---
        # 1. Preprocess y for TM (now fast and vectorized)
        y_tm_input = preprosses_TM_Decoder(y, N_NORMAL , num_bands=num_bands)
        s_int_tm = s_int_batch.cpu().numpy().astype(np.uint32)
        
        # 2. Fit the TM

       # if epoch == 10:
       #     LEARNING_RATE_AE = LEARNING_RATE_AE * 0.1

        if epoch % 2 == 0:

            tm_decoder.fit(y_tm_input, s_int_tm, epochs=1, incremental=True)

        else:

            # --- Encoder Training (via TM Feedback) ---
            # This section is now fully vectorized, replacing the BATCH_SIZE loop
            
          
            y_mega_batch = generate_perturbations_batch(y_tm_input, N_NORMAL, num_bands=num_bands)
        
            # 4. Get TM votes for all perturbations at once
            # Shape: [B * (1 + 2*N), M_classes]
            _, votesums_mega = tm_decoder.predict(y_mega_batch, return_class_sums=True) 

            # 5. Calculate scores for all perturbations (vectorized)
            # Shape: [B, 1 + 2*N]
            scores_batch = calculate_tm_scores_batch(votesums_mega, s_int_tm, M_NORMAL)

            # 6. Create the best "optimized" y for each instance (vectorized)
            # Shape: [B, N_features]
            y_optimized_thermo = create_best_variation_batch(y_tm_input, scores_batch, N_NORMAL, num_bands=num_bands)
            
            # 7. Reverse the preprocessing to get float values (vectorized)
            # Shape: [B, 2*N]
            y_optimized_recon_np = reverse_preprosses_TM_Decoder_Selector(y_optimized_thermo, N_NORMAL, num_bands=num_bands, mode='middle')

            # 8. Move target back to GPU


            y_optimized_reconstructed = (y_optimized_recon_np*LEARNING_RATE_AE + y*(1-LEARNING_RATE_AE))

           
        
            # 9. Compute loss between original y and optimized y

            x_reconstructed = y_optimized_reconstructed- noise

            x_tensor = torch.tensor(x_reconstructed, device=power.device, dtype=power.dtype)
            x_reconstructed = x_tensor * torch.sqrt(power + 1e-8)
            x_reconstructed = x_reconstructed.cpu().numpy()

            # Standerize x_reconstructed

            if epoch < STANDARDIZE_EPOCHS:
                x_reconstructed_mean = np.mean(x_reconstructed, axis=0, keepdims=True)
                x_reconstructed_std = np.std(x_reconstructed, axis=0, keepdims=True)
                x_reconstructed = (x_reconstructed - x_reconstructed_mean) / (x_reconstructed_std + 1e-8)
            
            for i in range(2*N_NORMAL):
                tm_encoders[i].fit(s_onehot_batch.cpu().numpy(), x_reconstructed[:,i], epochs=1, incremental=True)
            

        
        
       
        
        # --- Reporting ---
        if (epoch ) % 2 == 1:
            # TM BLER
            s_hat_int_tm = tm_decoder.predict(y_tm_input)
            bler_tm = (s_hat_int_tm != s_int_tm).mean()
            
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"BLER: {bler_tm:.9f}, "
                  f"SNR: {current_snr} dB, LR: {LEARNING_RATE_AE:.6f}")
    return tm_decoder, tm_encoders
# ==============================================================================
# --- NEW FUNCTION TO COMPARE DECODERS AND PLOT BLER ---
# ==============================================================================

def evaluate_and_plot_bler(encoder, decoder_tm, channel_type, n, m,
                           snr_min_db=-4, snr_max_db=8,
                           n_test_batches=100, batch_size=1000,
                           benchmark_data_list=None, # <-- MODIFIED to accept a list
                           save_path="bler_comparison_plot.png"):
    """
    Compares the Block Error Rate (BLER) of the TM decoder against
    a list of optional benchmarks and saves the plot.
    """
    print(f"\n--- Phase 2: Evaluating TM Decoder ({channel_type}) ---")

    snr_db_values = np.arange(snr_min_db, snr_max_db + 1, 1)
    bler_tm_list = []

    total_test_blocks = n_test_batches * batch_size
    print(f"Testing {total_test_blocks} blocks per SNR point from {snr_min_db} dB to {snr_max_db} dB...")

    start_time = time.time()
    for snr_db in snr_db_values:
        total_errors_tm = 0
        for _ in range(n_test_batches):
            s_int_batch = torch.randint(0, M_NORMAL, (batch_size,), device=DEVICE)
            s_onehot_batch = int_to_one_hot(s_int_batch, M_NORMAL)
            
            x = np.zeros((batch_size, 2*N_NORMAL))
            for i in range(2*N_NORMAL):
                x[:,i] = Normal_tm_encoders[i].predict(s_onehot_batch.cpu().numpy())

            x_tensor, power = power_normalize(torch.tensor(x, dtype=torch.float32))
            
            y_tensor, _ = channel(x_tensor, snr_db, k_bits=K_NORMAL, channel_type=channel_type)
            y = y_tensor.cpu().numpy()
        
            y_tm_input = preprosses_TM_Decoder(y, N_NORMAL , num_bands=num_bands)
            s_int_tm = s_int_batch.cpu().numpy().astype(np.uint32)
            s_hat_int_tm = decoder_tm.predict(y_tm_input)
            total_errors_tm += (s_hat_int_tm != s_int_tm).sum()

        bler_tm = total_errors_tm / total_test_blocks
        bler_tm_list.append(bler_tm)

        print(f"SNR: {snr_db:2d} dB | TM BLER: {bler_tm:1.9f}")
    
    end_time = time.time()
    print(f"TM evaluation finished in {end_time - start_time:.2f} seconds.")

    # ======================================================
    # --- MODIFIED PLOTTING SECTION ---
    # ======================================================
    plt.figure(figsize=(10, 7))
    
    # Plot 1: Your TM system
    plt.semilogy(snr_db_values, bler_tm_list, 'rs-', label='TM Encoder + TM Decoder (Learned)', markersize=8)
    
    # Plot 2: The benchmarks (if provided)
    if benchmark_data_list is not None:
        # Define some styles for the benchmarks
        styles = [('go--', 'o'), ('cv--', 'v')] # Green dashed, Cyan dashed
        
        for i, benchmark in enumerate(benchmark_data_list):
            style, marker = styles[i % len(styles)]
            plt.semilogy(
                benchmark['snr'], 
                benchmark['bler'], 
                style,
                label=benchmark['label'], 
                markersize=8
            )
    

    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Block Error Rate (BLER)', fontsize=14)
    plt.title(f'TM vs. Classical Benchmarks\n({CHANNEL_TYPE}, N={N_NORMAL}, K={K_NORMAL})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=1e-5, top=1.0)
    plt.xticks(snr_db_values)
    
    plt.savefig(save_path)
    print(f"BLER comparison plot saved to {os.path.abspath(save_path)}")
    plt.close()

    return snr_db_values, None, bler_tm_list

import matplotlib.pyplot as plt
import numpy as np

def plot_throughput_kbps(snr_db, bler_tm, benchmarks_list, k_bits, n_symbols, symbol_rate_hz=1_000_000, save_path="throughput_kbps.png"):
    """
    Plots Effective Throughput (Kbps) vs SNR.
    
    Args:
        symbol_rate_hz: Assumed speed of the channel. 
                        Default 1 MHz (1,000,000 symbols/sec).
                        Change this to match your target hardware speed.
    """
    print(f"\n--- Calculating Throughput (Assuming {symbol_rate_hz/1e6} Msps) ---")
    
    # 1. Calculate Maximum Raw Data Rate (in Kbps)
    # R = (4 bits / 8 symbols) * 1,000,000 = 500,000 bits/sec = 500 kbps
    bits_per_symbol = k_bits / n_symbols
    max_rate_kbps = (bits_per_symbol * symbol_rate_hz) / 1000.0
    
    print(f"Max Raw Rate: {max_rate_kbps} kbps")

    plt.figure(figsize=(10, 7))

    # --- 2. Calculate and Plot TM Throughput ---
    # Goodput = Max_Rate * (1 - BLER)
    # If BLER is 1 (all errors), Throughput is 0.
    # If BLER is 0 (perfect), Throughput is Max_Rate.
    throughput_tm = [max_rate_kbps * (1 - b) for b in bler_tm]
    
    plt.plot(snr_db, throughput_tm, 'rs-', label='TM System (Learned)', linewidth=2, markersize=8)

    # --- 3. Calculate and Plot Benchmarks ---
    if benchmarks_list:
        styles = [('go--', 'o'), ('cv--', 'v'), ('k-.', '.')]
        
        for i, benchmark in enumerate(benchmarks_list):
            if 'Finite' in benchmark['label']: continue # Skip theoretical bound if desired

            # Ensure lists are same length or interpolate (Assuming they share SNR axis for simplicity)
            # Using the benchmark's own BLER list
            tp_bench = [max_rate_kbps * (1 - b) for b in benchmark['bler']]
            
            style, marker = styles[i % len(styles)]
            plt.plot(benchmark['snr'], tp_bench, style, 
                     label=benchmark['label'], markersize=8)

    # --- Formatting ---
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Effective Throughput (kbps)', fontsize=14)
    plt.title(f'Throughput vs. SNR\n(Bandwidth: {symbol_rate_hz/1e6} MHz, Rate=0.5)', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(bottom=0, top=max_rate_kbps * 1.1) # Scale Y to show max rate
    
    plt.savefig(save_path)
    print(f"Throughput plot saved to {save_path}")
   



def evaluate_qpsk_repetition_bler(snr_min_db, snr_max_db, n_test_batches, batch_size, channel_type):
    """
    Evaluates the BLER of a (16, 4) Repetition Code with QPSK modulation.
    
    Configuration:
    - k = 4 info bits
    - (16, 4) Repetition Code (Rate 1/4) -> 16 coded bits (Repetition Factor = 4)
    - QPSK modulation (2 bits/symbol) -> 8 complex symbols (16 real dimensions)
    - Total Rate = k/symbols = 4/8 = 0.5 bits/symbol
    """
    print(f"\n--- Phase 2b: Evaluating QPSK + (16,4) Repetition Code ({channel_type}) ---")
    
    k = K_NORMAL        # 4 info bits
    n_coded_bits = 16   # Fixed for (16,4) code
    rep_factor = n_coded_bits // k # 4

    snr_db_values = np.arange(snr_min_db, snr_max_db + 1, 1)
    bler_qpsk_list = []
    
    total_test_blocks = n_test_batches * batch_size
    print(f"Testing {total_test_blocks} blocks per SNR point from {snr_min_db} dB to {snr_max_db} dB...")
    
    start_time = time.time()
    for snr_db in snr_db_values:
        total_errors = 0
        
        for _ in range(n_test_batches):
            # 1. Generate k info bits (0s and 1s)
            s_info_bits = np.random.randint(0, 2, (batch_size, k))
            
            # 2. Encode: (16, 4) Repetition Code
            # [b1, b2, b3, b4] -> [b1,b1,b1,b1, b2,b2,b2,b2, ...]
            # We repeat each element 4 times
            x_coded_bits = np.repeat(s_info_bits, rep_factor, axis=1) # Shape: (batch_size, 16)
            
            # 3. Modulate: QPSK
            # Map 0 -> +1, 1 -> -1 (BPSK mapping applied to I and Q separately)
            x_mapped = 1 - 2 * x_coded_bits # Shape: (batch_size, 16)
            
            # Create transmitted signal x with shape (batch_size, 16)
            # We treat the 16 items as 8 complex symbols (16 real values)
            # Even indices -> I component, Odd indices -> Q component
            x_tx = np.empty((batch_size, n_coded_bits), dtype=np.float32)
            x_tx[:, 0::2] = x_mapped[:, 0::2] # I components
            x_tx[:, 1::2] = x_mapped[:, 1::2] # Q components
            
            # Convert to tensor
            x_tensor = torch.tensor(x_tx, dtype=torch.float32).to(DEVICE)
            
            # 4. Channel
            # Note: k_bits is just for logging/normalization inside channel, passing K_NORMAL is fine
            y_tensor, _ = channel(x_tensor, snr_db, k_bits=K_NORMAL, channel_type=channel_type)
            y_received = y_tensor.cpu().numpy() # Shape: (batch_size, 16)
            
            # 5. Demodulate: Hard-decision QPSK
            # Received > 0 -> mapped +1 -> original bit 0
            # Received < 0 -> mapped -1 -> original bit 1
            y_coded_bits = np.zeros((batch_size, n_coded_bits), dtype=int)
            
            # Check I components (even indices) and Q components (odd indices)
            y_coded_bits[:, 0::2] = (y_received[:, 0::2] < 0).astype(int)
            y_coded_bits[:, 1::2] = (y_received[:, 1::2] < 0).astype(int)
            
            # 6. Decode: Majority vote for (16, 4) repetition code
            decoded_info_bits = np.empty((batch_size, k), dtype=int)
            
            for i in range(k):
                # Extract the block of 4 repeated bits corresponding to info bit i
                # e.g., for i=0, we take indices 0,1,2,3
                start_idx = i * rep_factor
                end_idx = (i + 1) * rep_factor
                bit_block = y_coded_bits[:, start_idx : end_idx]
                
                # Sum the 1s
                vote_sum = np.sum(bit_block, axis=1) # Shape: (batch_size,)
                
                # Majority vote: If sum >= 2, we assume it was a 1. 
                # (Note: Ties (2 vs 2) need a decision. Here we define >= 2 as 1, or > 2 as 1.
                # Standard repetition code usually breaks ties randomly or consistently.
                # Here: > 2 means strictly 3 or 4 votes.
                decoded_info_bits[:, i] = (vote_sum > 2).astype(int)
            
            # 7. Calculate BLER
            # A block error occurs if *any* of the k info bits are wrong
            errors = np.any(decoded_info_bits != s_info_bits, axis=1)
            total_errors += np.sum(errors)

        bler = total_errors / total_test_blocks
        bler_qpsk_list.append(bler)
        print(f"SNR: {snr_db:2d} dB | QPSK Rep. BLER: {bler:1.9f}")
        
    end_time = time.time()
    print(f"QPSK Rep. evaluation finished in {end_time - start_time:.2f} seconds.")
    
    return snr_db_values, np.array(bler_qpsk_list)





def evaluate_hamming_bler(snr_min_db, snr_max_db, n_test_batches, batch_size, channel_type):
    """
    Evaluates the BLER of an (8, 4) Extended Hamming Code with BPSK modulation.
    """
    print(f"\n--- Phase 2c: Evaluating BPSK + (8,4) Ext. Hamming Code ({channel_type}) ---")
    
    k = 4 # Fixed for (8,4) Hamming
    n_coded_bits = 8 # Fixed for (8,4) Hamming
    
    # BPSK means 1 bit per symbol, so we need 8 symbols for 8 bits
    n_symbols = n_coded_bits 

    snr_db_values = np.arange(snr_min_db, snr_max_db + 1, 1)
    bler_hamming_list = []
    
    total_test_blocks = n_test_batches * batch_size
    print(f"Testing {total_test_blocks} blocks per SNR point from {snr_min_db} dB to {snr_max_db} dB...")
    
    start_time = time.time()
    for snr_db in snr_db_values:
        total_errors = 0
        
        for _ in range(n_test_batches):
            # 1. Generate k=4 info bits
            s_info_bits = np.random.randint(0, 2, (batch_size, k))
            d1, d2, d3, d4 = s_info_bits[:, 0], s_info_bits[:, 1], s_info_bits[:, 2], s_info_bits[:, 3]
            
            # 2. Encode: (8,4) Extended Hamming Code
            # (7,4) Parity bits
            p1 = (d1 + d2 + d4) % 2
            p2 = (d1 + d3 + d4) % 2
            p3 = (d2 + d3 + d4) % 2
            
            # (7,4) codeword structure: [p1, p2, d1, p3, d2, d3, d4]
            c7 = np.stack([p1, p2, d1, p3, d2, d3, d4], axis=1)
            
            # (8,4) Overall parity bit (for the extended code)
            p4 = np.sum(c7, axis=1) % 2
            
            # Final 8-bit codeword: [p1, p2, d1, p3, d2, d3, d4, p4]
            x_coded_bits = np.hstack([c7, p4[:, np.newaxis]]) # Shape: (batch_size, 8)
            
            # 3. Modulate: BPSK
            # Map 0 -> +1, 1 -> -1
            x_mapped = 1 - 2 * x_coded_bits # Shape: (batch_size, 8)
            
            # Create transmitted signal x.
            # We need 2 real values per symbol (I and Q).
            # Total real dimensions = 2 * n_symbols = 16
            x_tx_complex = np.zeros((batch_size, 2 * n_symbols), dtype=np.float32)
    
            x_tx_complex[:, 0::2] = x_mapped
            
            x_tensor = torch.tensor(x_tx_complex, dtype=torch.float32).to(DEVICE)
            
            # 4. Channel
            y_tensor, _ = channel(x_tensor, snr_db, k_bits=K_NORMAL, channel_type=channel_type)
            y_received = y_tensor.cpu().numpy() # Shape: (batch_size, 16)
            
            # 5. Demodulate: Hard-decision BPSK from I-channel
            y_I = y_received[:, 0::2] # (batch_size, 8)
            y_coded_bits = (y_I < 0).astype(int)
            
            # 6. Decode: Hard-decision (8,4) Hamming Decoder
            r = y_coded_bits
            p1,p2,d1,p3,d2,d3,d4,p4 = r[:,0],r[:,1],r[:,2],r[:,3],r[:,4],r[:,5],r[:,6],r[:,7]
            
            # Recalculate syndromes
            s1 = (p1 + d1 + d2 + d4) % 2
            s2 = (p2 + d1 + d3 + d4) % 2
            s3 = (p3 + d2 + d3 + d4) % 2
            
            # Syndrome vector (s3, s2, s1) points to error position 1-7
            syndrome_int = s1 * 1 + s2 * 2 + s3 * 4
            
            # Overall parity check
            sP = np.sum(r, axis=1) % 2
            
            corrected_bits = r.copy()
            
            # Create masks for correction logic
            correctable_mask = (syndrome_int != 0) & (sP == 1)
            p4_error_mask = (syndrome_int == 0) & (sP == 1)

            # Apply corrections
            # Using a loop for clarity on specific bit flipping
            for i in range(batch_size):
                s = syndrome_int[i]
                if correctable_mask[i]:
                    # Correct single error at position s-1
                    corrected_bits[i, s-1] = 1 - corrected_bits[i, s-1]
                elif p4_error_mask[i]:
                    # Correct error in p4 (bit 7)
                    corrected_bits[i, 7] = 1 - corrected_bits[i, 7]
                
            # 7. Extract info bits: positions [2, 4, 5, 6] correspond to d1, d2, d3, d4
            decoded_info_bits = corrected_bits[:, [2, 4, 5, 6]] 
            
            # 8. Calculate BLER
            errors = np.any(decoded_info_bits != s_info_bits, axis=1)
            total_errors += np.sum(errors)

        bler = total_errors / total_test_blocks
        bler_hamming_list.append(bler)
        print(f"SNR: {snr_db:2d} dB | BPSK Hamming BLER: {bler:1.9f}")
        
    end_time = time.time()
    print(f"BPSK Hamming evaluation finished in {end_time - start_time:.2f} seconds.")
    
    return snr_db_values, np.array(bler_hamming_list)




# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == '__main__':

    # --- 1. Initialize TM Encoder and Decoder ---
    Normal_tm_decoder = TMCoalescedClassifier(
        number_of_clauses=16800,
        T=int(0.8108165710735751 * 16800),
        s=6.0,
        max_included_literals=65,
        platform="CPU",
        weighted_clauses=False,
        type_iii_feedback=False
    )

    Normal_tm_encoders = []
    for i in range(N_NORMAL*2):
        tm_enc = TMRegressor(
            600,
            int(0.8171086283457829 * 600),
            9.0,
            platform="CPU",
            max_included_literals=7,
            weighted_clauses=False,
        )
        Normal_tm_encoders.append(tm_enc)
    
    # --- 2. Train the TM system ---
    Normal_tm_decoder, Normal_tm_encoders = train_autoencoder(Normal_tm_decoder,Normal_tm_encoders,
                          epochs=EPOCHS_AE,
                          snr_db=AE_TRAIN_SNR_DB,
                          channel_type=CHANNEL_TYPE)




    # 2. Generate SNR range
    snr_range = np.linspace(-4, 10, 200)

 



    # Package the benchmark data into a list
    benchmarks_list = [
  

    ]

    

    # --- 4. Evaluate the TM System and Plot Comparison ---
    evaluate_and_plot_bler_output = evaluate_and_plot_bler(
        encoder=Normal_tm_encoders,
        decoder_tm=Normal_tm_decoder,
        channel_type=CHANNEL_TYPE,
        n=N_NORMAL,
        m=M_NORMAL,
        snr_min_db=SNR_MIN,
        snr_max_db=SNR_MAX,
        n_test_batches=N_BATCHES,
        batch_size=B_SIZE,
        benchmark_data_list=benchmarks_list, 
        save_path="bler_TM_vs_Benchmarks.png"
    )

    plot_throughput_kbps(
    snr_db=np.arange(SNR_MIN, SNR_MAX + 1, 1),
    bler_tm=evaluate_and_plot_bler_output[2],  
    benchmarks_list=benchmarks_list,
    k_bits=K_NORMAL,       
    n_symbols=N_NORMAL,   
    symbol_rate_hz=2_000_000,
    save_path="throughput_kbps_comparison_AWGN.png"
)

 
    os.makedirs('models', exist_ok=True)

  
    joblib.dump(Normal_tm_decoder, 'models/tm_decoder_model.joblib')
    for i, tm_enc in enumerate(Normal_tm_encoders):
        joblib.dump(tm_enc, f'models/tm_encoder_model_{i}.joblib')
