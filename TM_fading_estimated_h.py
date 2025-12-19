import os
# THESE LINES MUST BE FIRST, before tmu or anything else
os.environ['NUMBA_DEBUG'] = '0'
os.environ['NUMBA_DISABLE_DEBUG_LOGGING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from numba import njit
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.models.regression.vanilla_regressor import TMRegressor
import logging

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import time
from scipy import integrate # <--- ADDED THIS
import joblib # Added for path handling



model_decoder = 'normal_decoder.joblib'
model_encoder_prefix = 'normal_encoder_'

@njit(debug=False)
def preprosses_TM_Decoder(y_np, n, num_bands=5, min_val=-4.0, max_val=4.0):
    """
    Vectorized preprocessing of the received signal y for TM Classifier.
    Assumes y is in the range [min_val, max_val].
    """
    batch_size = y_np.shape[0]
    num_features = 2 * n
    
    # Calculate span
    span = max_val - min_val

    # 1. Scale, round, and clamp
    # Map [min_val, max_val] to [0, num_bands]
    # Original: val_scaled = (y_np + 3.0) * (num_bands / 6.0)
    # New:      val_scaled = (y_np - min_val) * (num_bands / span)
    val_scaled = (y_np - min_val) * (num_bands / span)
    
    val_int_f = np.empty_like(val_scaled)
    np.round(val_scaled, 0, val_int_f)
    val_int = val_int_f.astype(np.int32)
    
    # Clamp value between 0 and num_bands - 1
    val = np.clip(val_int, 0, num_bands - 1) 

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

import numpy as np
from numba import njit


def revert_fading(y, h, epsilon=1e-8):
    batch_size = y.shape[0]
    n_symbols = y.shape[1] // 2
    y_reshaped = y.reshape(batch_size, n_symbols, 2)
    y_complex = torch.view_as_complex(y_reshaped.contiguous())
    h_complex = torch.view_as_complex(h.contiguous())

    h_conj = torch.conj(h_complex)
    h_mag_sq = h_complex.abs().pow(2)
    h_mag_sq_safe = h_mag_sq + epsilon
    h_inv = h_conj / h_mag_sq_safe
    h_inv_unsq = h_inv.unsqueeze(1)

    x_complex = y_complex * h_inv_unsq
    x_real = torch.view_as_real(x_complex)
    return x_real.reshape(batch_size, -1)

@njit(debug=False)
def calculate_h_scores_batch(votesums_mega_batch, s_int_batch, num_variations):

    """
    Calculates scores specifically for h-parameter perturbations.
    votesums_mega_batch: (B * num_variations, M)
    """
    batch_size = s_int_batch.shape[0]
    
    # 1. Repeat correct class labels for the mega batch
    # Shape: [B * num_variations]
    correct_class_repeated = np.empty(batch_size * num_variations, dtype=np.int32)
    for i in range(batch_size):
        for j in range(num_variations):
            correct_class_repeated[i * num_variations + j] = s_int_batch[i]

    # 2. Extract votes for the correct class
    num_rows = votesums_mega_batch.shape[0]
    correct_class_votes = np.empty(num_rows, dtype=votesums_mega_batch.dtype)
    
    for i in range(num_rows):
        idx = correct_class_repeated[i]
        correct_class_votes[i] = votesums_mega_batch[i, idx]

    # 3. Calculate Total Votes
    total_votes = np.sum(votesums_mega_batch, axis=1)
    
    # 4. Calculate Score (Lower is better)
    # Score = Total_Votes - 2 * Correct_Class_Votes
    scores = total_votes - 2 * correct_class_votes
    
    # Reshape to (Batch, Num_Variations)
    return scores.reshape(batch_size, num_variations)

def estimate_h_from_x_y(x, y, epsilon=1e-8):
    """
    Estimates the channel coefficient h given the transmitted signal x 
    and the received signal y using Least Squares.
    
    Robustly handles mixtures of Numpy arrays and Torch Tensors (CPU/GPU).
    """
    # 1. Ensure inputs are Tensors
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()
        
    # 2. Ensure they are on the same device (move y to x's device)
    y = y.to(x.device)

    batch_size = x.shape[0]
    n_symbols = x.shape[1] // 2
    
    # 3. Convert to Complex Tensors
    x_reshaped = x.reshape(batch_size, n_symbols, 2)
    x_complex = torch.view_as_complex(x_reshaped.contiguous())
    
    y_reshaped = y.reshape(batch_size, n_symbols, 2)
    y_complex = torch.view_as_complex(y_reshaped.contiguous())
    
    # 4. Compute Least Squares Estimate: h = sum(y * conj(x)) / sum(|x|^2)
    numerator = torch.sum(y_complex * torch.conj(x_complex), dim=1)
    denominator = torch.sum(x_complex.abs().pow(2), dim=1) + epsilon
    
    h_complex = numerator / denominator
    
    # 5. Convert back to Real representation (Batch, 2)
    h_real = torch.view_as_real(h_complex)
    
    # Return numpy array (move to CPU first)
    return h_real.detach().cpu().numpy()

# --- Configuration ---
CHANNEL_TYPE = 'Rayleigh'  # Options: 'AWGN', 'Rayleigh', 'Rician'

# Autoencoder (Normal Communication) Parameters
N_NORMAL = 7
K_NORMAL = 4
M_NORMAL = 2**K_NORMAL





# Training Parameters
BATCH_SIZE = 60 
LEARNING_RATE_AE_decoder = 0.29115932906326103
LEARNING_RATE_AE_param = 0.1690294459533964 
EPOCHS_AE = 20000 
num_bands = 50 
STANDARDIZE_EPOCHS = 100000 
STANDARDIZE_EPOCHS_PARAM = 20
min_val_encoder = -3
max_val_encoder = 3
min_val_param = -3
max_val_param = 3

mode = 'lower'  # Options: 'lower', 'upper', 'middle', 'uniform', 'gaussian'

# Define evaluation parameters
SNR_MIN = -20
SNR_MAX = 20
N_BATCHES = 10
B_SIZE = 10000

if CHANNEL_TYPE == 'd':
    AE_TRAIN_SNR_DB = 6  # This is not in the new params, but kept from original
    TRAIN_SNR_DB_MIN = -2
    TRAIN_SNR_DB_MAX = 8
else:  # Rayleigh or Rician
    AE_TRAIN_SNR_DB = 8
    TRAIN_SNR_DB_MIN = -1.5 
    TRAIN_SNR_DB_MAX = 32 

print(f"Autoencoder training SNR set to {TRAIN_SNR_DB_MIN} to {TRAIN_SNR_DB_MAX} dB for {CHANNEL_TYPE} channel.")

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
# --- Training Functions ---
def train_autoencoder(tm_decoder, tm_encoders, param_estimator, epochs, snr_db, channel_type):
    """
    Trains the DNN autoencoder (Encoder + Decoder) AND the TM Decoder
    simultaneously on the same channel data.
    param_estimator: list of TMRegressor instances for parameter estimation
    Refactored to be efficient by vectorizing the TM feedback loop.
    
    NOTE: This implementation follows the original code's logic, which *only*
    trains the ENCODER using the TM feedback loss. The DNN DECODER is
    *not* trained.
    """
    print(f"\n--- Phase 1: Pre-training Autoencoder and TM for {channel_type} ---")

    # Number of bands for thermometer encoding
    global LEARNING_RATE_AE_decoder
    # This optimizer *only* trains the encoder, per the original code's logic

    s_int_batch = torch.randint(0, M_NORMAL, (BATCH_SIZE,), device=DEVICE)
    s_onehot_batch = int_to_one_hot(s_int_batch, M_NORMAL)
    s_onehot_batch = s_onehot_batch.cpu().numpy()
    yyy = np.array([min_val_encoder]*8 + [max_val_encoder]*8)
    yyy_param = np.array([min_val_param]*8 + [max_val_param]*8)

    best_validation_bler = 1.0

    momentum = 0.14290201892644683
    running_mean = None
    running_std = None
    running_bler = 1.0

    print(yyy)
    for tm_encoder in tm_encoders:
        tm_encoder.init_after(X= s_onehot_batch, Y= yyy)
        tm_encoder.init_clause_bank(X= s_onehot_batch, Y= np.zeros((16,16)))
        tm_encoder.init_weight_bank(X= s_onehot_batch, Y= np.zeros((16,16)))

    for param_est in param_estimator:
        param_est.init_after(X= s_onehot_batch, Y= yyy_param)
        param_est.init_clause_bank(X= s_onehot_batch, Y= np.zeros((16,16)))
        param_est.init_weight_bank(X= s_onehot_batch, Y= np.zeros((16,16)))



    for epoch in range(epochs+1):
        s_int_batch = torch.randint(0, M_NORMAL, (BATCH_SIZE,), device=DEVICE)
        s_onehot_batch = int_to_one_hot(s_int_batch, M_NORMAL)

        # Generate one SNR per SAMPLE in the batch
        current_snr_db_batch = np.random.rand(BATCH_SIZE) * (TRAIN_SNR_DB_MAX - TRAIN_SNR_DB_MIN) + TRAIN_SNR_DB_MIN # <--- MODIFIED
        snr_linear_batch = 10**(current_snr_db_batch / 10.0) # Shape: [BATCH_SIZE] # <--- MODIFIED
        
        # Store the average SNR for reporting
        current_snr = current_snr_db_batch.mean() # <--- MODIFIED
        
        

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # --- Forward Pass ---
        x = np.zeros((BATCH_SIZE, 2*N_NORMAL))
        for i in range(2*N_NORMAL):
            x[:,i] = tm_encoders[i].predict(s_onehot_batch.cpu().numpy())

        

        x, power = power_normalize(torch.tensor(x))
        x_org = x.clone()
    

        # --- fading 
        h = None

        if channel_type in ['Rayleigh', 'Rician']:
            if h is None:
                
                h = generate_fading_coefficient(x.shape[0], channel_type, x.device)
                
                signal_after_fading = apply_fading(x, h)
                signal_after_fading = signal_after_fading.cpu().numpy()
              
              #  x = signal_after_fading
        else:
            signal_after_fading = x
            x = signal_after_fading
            signal_after_fading = signal_after_fading.cpu().numpy()# <--- Make sure x is the numpy array
            h = None
                

        



        rate = K_NORMAL / N_NORMAL 

        
        signal_power_batch = np.mean(x_org.numpy()**2, axis=1) # Shape: [BATCH_SIZE]

        ebn0_linear_batch = 10**(current_snr_db_batch / 10.0) 
        snr_linear_batch = ebn0_linear_batch * rate 

        noise_variance_batch = 1.0 / (snr_linear_batch + 1e-10)

        # Create mask for samples with valid power
        valid_power_mask = signal_power_batch > 1e-10

     
        noise_variance_batch[valid_power_mask] = signal_power_batch[valid_power_mask] / snr_linear_batch[valid_power_mask]
            
        # 5. Generate and Scale Noise
        noise = torch.randn_like(torch.tensor(x.numpy())).numpy() # Shape: [BATCH_SIZE, 2*N_NORMAL]
        
        # Get per-sample noise standard deviation and reshape for broadcasting
        noise_std_batch_col = np.sqrt(noise_variance_batch)[:, np.newaxis] # Shape: [BATCH_SIZE, 1]
        
        # Scale noise per-sample
        noise_scaled = noise * noise_std_batch_col
        
        y = signal_after_fading + noise_scaled
    
        # --- TM Training ---
        # 1. Preprocess y for TM (now fast and vectorized)
        y_tm_input_param = preprosses_TM_Decoder(y, N_NORMAL , num_bands=num_bands)
        s_int_tm = s_int_batch.cpu().numpy().astype(np.uint32)
        
        # estimate h
        h_est = np.zeros((BATCH_SIZE, 2))
        for i in range(2):
            h_est[:,i] = param_estimator[i].predict(y_tm_input_param)


     

        y_estimated = revert_fading(torch.tensor(y, device=DEVICE, dtype=torch.float32), torch.tensor(h_est, device=DEVICE, dtype=torch.float32))
        y_estimated = y_estimated.cpu().numpy()

        y_tm_input = preprosses_TM_Decoder(y_estimated, N_NORMAL , num_bands=num_bands)


        


        if epoch % 3 == 0:

            
            tm_decoder.fit(y_tm_input, s_int_tm, epochs=1, incremental=True)

        elif epoch % 3 == 1:
           

            
            h_targets = estimate_h_from_x_y(x=x_org, y=y)
            
 
            param_estimator[0].fit(y_tm_input_param, h_targets[:, 0], epochs=1, incremental=True)
            param_estimator[1].fit(y_tm_input_param, h_targets[:, 1], epochs=1, incremental=True)




            



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
            y_optimized_recon_np = reverse_preprosses_TM_Decoder_Selector(y_optimized_thermo, N_NORMAL, num_bands=num_bands, mode=mode)

    

            if channel_type in ['Rayleigh', 'Rician']:
                y_optimized_reconstructed_tensor = torch.tensor(y_optimized_recon_np, device=power.device, dtype=power.dtype)
                # revert revert fading 
                y_optimized_reconstructed_tensor = apply_fading(y_optimized_reconstructed_tensor, torch.tensor(h_est, device=power.device, dtype=power.dtype))
                
              #  y_optimized_reconstructed_tensor = y_optimized_reconstructed_tensor - noise_scaled 

                # esto,ate h from estimated to original
                h_true_estimmated = estimate_h_from_x_y(x=x_org, y=y)

                h_true_estimmated = torch.tensor(h_true_estimmated, device=power.device, dtype=power.dtype)

                y_optimized_reconstructed_tensor = revert_fading(y_optimized_reconstructed_tensor, h_true_estimmated)
                
                
                y_optimized_reconstructed_tensor = y_optimized_reconstructed_tensor

                y_optimized_reconstructed = y_optimized_reconstructed_tensor.cpu().numpy()
            else:
                y_optimized_reconstructed = y_optimized_reconstructed - noise_scaled 
                y_optimized_reconstructed = y_optimized_reconstructed
            # 9. Compute loss between original y and optimized y

            
            
            x_tensor = torch.tensor(y_optimized_reconstructed, device=power.device, dtype=power.dtype)
            x_tensor = (x_tensor*LEARNING_RATE_AE_decoder + x_org*(1-LEARNING_RATE_AE_decoder))
            x_reconstructed = x_tensor * torch.sqrt(power + 1e-8)
            x_reconstructed = x_reconstructed.cpu().numpy()

            # Standerize x_reconstructed

            if epoch < STANDARDIZE_EPOCHS:
                # 1. Calculate stats for CURRENT batch
                batch_mean = np.mean(x_reconstructed, axis=0) # Shape: (features,)
                batch_std = np.std(x_reconstructed, axis=0)   # Shape: (features,)

                # 2. Update Rolling Stats
                if running_mean is None:
                    # Initialize
                    running_mean = batch_mean
                    running_std = batch_std
                else:
                    # Update with momentum
                    running_mean = momentum * running_mean + (1 - momentum) * batch_mean
                    running_std = momentum * running_std + (1 - momentum) * batch_std

                # 3. Apply Standardization using ROLLING stats
                # Using rolling stats prevents large jumps in gradients
                x_reconstructed = (x_reconstructed - running_mean) / (running_std + 1e-8)
            
            for i in range(2*N_NORMAL):
                tm_encoders[i].fit(s_onehot_batch.cpu().numpy(), x_reconstructed[:,i], epochs=1, incremental=True)
            
           # print("After encoder training epoch:", epoch)
            

        
        
       
        
        # --- Reporting ---
        if (epoch) % 10 == 1:
            # Existing Reporting Code
            s_hat_int_tm = tm_decoder.predict(y_tm_input)
            current_bler = (s_hat_int_tm != s_int_tm).mean()

            if running_bler is None:
                running_bler = current_bler
            else:
                running_bler = 0.9 * running_bler + (1 - 0.9) * current_bler

            # difrence between h and h_est
           
            
       
            
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Curr BLER: {current_bler:.5f}, "
                  f"Run BLER: {running_bler:.5f}, "
                  f"Avg SNR: {current_snr:.2f} dB"
             )
            
            
            if current_bler < 0.15:
                    

                # =======================================================
                # NEW: SAVE BEST MODEL LOGIC (Fixed SNR Validation)
                # =======================================================
                # We test at a fixed SNR (e.g., 5dB) to ensure fair comparison
                val_snr_db = 20.0 
                val_batch_size = 10000 # Larger batch for statistical accuracy
                
                # 1. Generate Validation Data
                s_val_int = torch.randint(0, M_NORMAL, (val_batch_size,), device=DEVICE)
                s_val_onehot = int_to_one_hot(s_val_int, M_NORMAL)
                
                # Predict X using encoders
                x_val = np.zeros((val_batch_size, 2*N_NORMAL))
                for i in range(2*N_NORMAL):
                    x_val[:,i] = tm_encoders[i].predict(s_val_onehot.cpu().numpy())
                
                x_val_tensor, power_val = power_normalize(torch.tensor(x_val))
                
                # 2. Channel at FIXED SNR (No Randomness!)
                y_val_tensor, _ = channel(x_val_tensor, val_snr_db, K_NORMAL, channel_type)
                y_val = y_val_tensor.cpu().numpy()
                
                # 3. Preprocess and Decode
                y_val_tm = preprosses_TM_Decoder(y_val, N_NORMAL, num_bands=num_bands)

                # estimate h for validation set
                h_val_est = np.zeros((val_batch_size, 2))
                for i in range(2):
                    h_val_est[:,i] = param_estimator[i].predict(y_val_tm)
                
                y_val_estimated = revert_fading(torch.tensor(y_val, device=DEVICE, dtype=torch.float32), torch.tensor(h_val_est, device=DEVICE, dtype=torch.float32))
                y_val_estimated = y_val_estimated.cpu().numpy()

                y_val_tm = preprosses_TM_Decoder(y_val_estimated, N_NORMAL , num_bands=num_bands)

                
                # (Optional: Include param estimation in validation if needed, 
                # usually simpler to skip for quick model selection)
                
                s_hat_val = tm_decoder.predict(y_val_tm)
                
                # 4. Calculate Validation BLER
                val_bler = (s_hat_val != s_val_int.cpu().numpy()).sum() / val_batch_size
                
                # 5. Save if Best
                if val_bler < best_validation_bler:
                    best_validation_bler = val_bler
                    print(f"   >>> New Best Model Found! (Val BLER: {best_validation_bler:.5f} @ {val_snr_db}dB)")
                    
                    joblib.dump(tm_decoder, f'models/best_tm_decoder.joblib')
                    for i, enc in enumerate(tm_encoders):
                        joblib.dump(enc, f'models/best_tm_encoder_{i}.joblib')
                    for i, pest in enumerate(param_estimator):
                        joblib.dump(pest, f'models/best_param_est_{i}.joblib')
                # =======================================================

    # load best model before returning
    tm_decoder = joblib.load(f'models/best_tm_decoder.joblib')
    tm_encoders = []
    for i in range(2*N_NORMAL):
        enc = joblib.load(f'models/best_tm_encoder_{i}.joblib')
        tm_encoders.append(enc)
    param_estimators = []
    for i in range(2):
        pest = joblib.load(f'models/best_param_est_{i}.joblib')
        param_estimators.append(pest)
    
    return tm_decoder, tm_encoders, param_estimators
# ==============================================================================
# --- NEW FUNCTION TO COMPARE DECODERS AND PLOT BLER ---
# ==============================================================================

def evaluate_and_plot_bler(encoder, decoder_tm, Normal_param_estimator, channel_type, n, m, # not used for plot in report 
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
                x[:,i] = encoder[i].predict(s_onehot_batch.cpu().numpy())

            x_tensor, power = power_normalize(torch.tensor(x))
            
            y_tensor, _ = channel(x_tensor, snr_db, K_NORMAL, channel_type)
            y = y_tensor.cpu().numpy()
        
            y_tm_input = preprosses_TM_Decoder(y, N_NORMAL , num_bands=num_bands)

            # parameter estimation
            h_est = np.zeros((batch_size, 2))
            for i in range(2):
                h_est[:,i] = Normal_param_estimator[i].predict(y_tm_input)
            y_estimated = revert_fading(torch.tensor(y, device=DEVICE, dtype=torch.float32), torch.tensor(h_est, device=DEVICE, dtype=torch.float32))
            y_estimated = y_estimated.cpu().numpy()
            y_tm_input = preprosses_TM_Decoder(y_estimated, N_NORMAL , num_bands=num_bands)

            s_int_tm = s_int_batch.cpu().numpy().astype(np.uint32)
            s_hat_int_tm = decoder_tm.predict(y_tm_input)
            total_errors_tm += (s_hat_int_tm != s_int_tm).sum()

        bler_tm =  total_errors_tm / total_test_blocks
        bler_tm_list.append(bler_tm)

        print(f"SNR: {snr_db:2d} dB | TM BLER: {bler_tm:1.9f}")
    
    end_time = time.time()
    print(f"TM evaluation finished in {end_time - start_time:.2f} seconds.")

    # ======================================================
    # --- MODIFIED PLOTTING SECTION ---
    # ======================================================
    plt.figure(figsize=(10, 7))
    
    # Plot 1: Your TM system
    objective_value = integrate.trapezoid(bler_tm_list, snr_db_values) # <--- FIXED

    print(f"Objective Value (Area under TM BLER curve): {objective_value:.6f}")

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
    
    # Add Shannon Limit for R=0.5
   # plt.axvline(x=-3.83, color='b', linestyle=':', label='Shannon Limit (R=0.5) @ -3.83 dB')
    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Block Error Rate (BLER)', fontsize=14)
    plt.title(f'TM vs. Classical Benchmarks\n({CHANNEL_TYPE}, N={N_NORMAL}, K={K_NORMAL})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #plt.ylim(bottom=1e-3, top=1.0)
    plt.xticks(snr_db_values)
    
    plt.savefig(save_path)
    print(f"BLER comparison plot saved to {os.path.abspath(save_path)}")
    plt.close()

    return snr_db_values, None, bler_tm_list

def get_classical_benchmarks(snr_range, batch_size=10000, num_batches=10, channel_type='Rayleigh', device=torch.device('cpu')):
    """
    Simulates classical benchmarks:
    1. Pilot-Aided Hamming(7,4): N=8 (1 Pilot + 7 Code). K=4.
    2. Differential BPSK (DPSK): N=9 (1 Ref + 8 Data). K=8.
    
    Returns:
        dict: Keys are 'pilot_hamming' and 'dpsk'.
    """
    print(f"\n--- Simulating Blind/Practical Benchmarks ({channel_type}) ---")
    
    # --- Shared: Hamming (7,4) Generator Matrix ---
    G = torch.tensor([
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1]
    ], dtype=torch.float32, device=device)

    # Generate 16 Codewords for Hamming
    nums = torch.arange(16, device=device).unsqueeze(1)
    messages = (nums >> torch.arange(3, -1, -1, device=device)) & 1
    messages = messages.float()
    codewords_bits = torch.matmul(messages, G) % 2 
    codewords_bpsk = 2 * codewords_bits - 1 

    results = {'pilot_hamming': [], 'dpsk': []}

    for snr in snr_range:
        err_pilot = 0
        err_dpsk = 0
        
        for _ in range(num_batches):
            # ===========================================================
            # Method 1: Pilot-Aided Hamming (7,4) [K=4]
            # ===========================================================
            indices = torch.randint(0, 16, (batch_size,), device=device)
            chosen_codewords = codewords_bpsk[indices] 
            pilots = torch.ones((batch_size, 1), device=device)
            tx_block_real = torch.cat((pilots, chosen_codewords), dim=1) # (B, 8)
            
            tx_complex_fmt = torch.zeros(batch_size, 16, device=device)
            tx_complex_fmt[:, 0::2] = tx_block_real
            
            # Channel (K=4)
            y_rx_flat, _ = channel(tx_complex_fmt, snr, 4, channel_type)
            y_rx_c = torch.view_as_complex(y_rx_flat.reshape(batch_size, 8, 2).contiguous())
            
            # Receiver
            h_est = y_rx_c[:, 0].unsqueeze(1) 
            y_data = y_rx_c[:, 1:] 
            
            codebook_flat = torch.zeros(16, 14, device=device)
            codebook_flat[:, 0::2] = codewords_bpsk
            codebook_c = torch.view_as_complex(codebook_flat.reshape(16, 7, 2).contiguous()).unsqueeze(0)
            
            candidates = h_est.unsqueeze(1) * codebook_c 
            dist = torch.sum(torch.abs(y_data.unsqueeze(1) - candidates)**2, dim=2)
            pred_pilot = torch.argmin(dist, dim=1)
            err_pilot += (pred_pilot != indices).sum().item()

            # ===========================================================
            # Method 2: Differential BPSK (DPSK) [K=8]
            # ===========================================================
            dpsk_bits = torch.randint(0, 2, (batch_size, 8), device=device).float()
            
            # Tx Array: (B, 9)
            tx_dpsk = torch.zeros(batch_size, 9, device=device)
            tx_dpsk[:, 0] = 1.0 # Reference
            
            mod_data = 1.0 - 2.0 * dpsk_bits 
            curr_sym = torch.ones(batch_size, device=device)
            for k in range(8): 
                curr_sym = curr_sym * mod_data[:, k]
                tx_dpsk[:, k+1] = curr_sym
                
            tx_dpsk_flat = torch.zeros(batch_size, 18, device=device)
            tx_dpsk_flat[:, 0::2] = tx_dpsk
            
            # Channel (K=8)
            y_dpsk_flat, _ = channel(tx_dpsk_flat, snr, 8, channel_type)
            y_dpsk_c = torch.view_as_complex(y_dpsk_flat.reshape(batch_size, 9, 2).contiguous())
            
            # Detection
            detected_bits = torch.zeros_like(dpsk_bits)
            for k in range(8):
                metric = (y_dpsk_c[:, k+1] * torch.conj(y_dpsk_c[:, k])).real
                detected_bits[:, k] = torch.where(metric < 0, 1.0, 0.0)
                
            errors_per_block = (detected_bits != dpsk_bits).sum(dim=1)
            err_dpsk += (errors_per_block > 0).sum().item()

        results['pilot_hamming'].append(err_pilot / (num_batches * batch_size))
        results['dpsk'].append(err_dpsk / (num_batches * batch_size))
        
        print(f"SNR {snr:2d} | Pilot-Hamming: {results['pilot_hamming'][-1]:.5f} | DPSK: {results['dpsk'][-1]:.5f}")

    # RETURN A DICTIONARY
    return {
        'pilot_hamming': {'snr': snr_range, 'bler': np.array(results['pilot_hamming']), 'label': 'Pilot-Aided Hamming (Practical)'},
        'dpsk': {'snr': snr_range, 'bler': np.array(results['dpsk']), 'label': 'Differential BPSK (Blind)'},
    }

def get_style_for_label(label):
    """
    Returns (color, marker, linestyle) based on the label content
    to ensure consistent plotting across different graphs.
    """
    label_lower = label.lower()
    
    if "tm" in label_lower:
        return 'r', 's', '-'  # Red Squares, Solid (TM)
    elif "pilot" in label_lower:
        return 'g', 'o', '--' # Green Circles, Dashed (Pilot Hamming)
    elif "dpsk" in label_lower:
        return 'c', 'v', '--' # Cyan Triangles, Dashed (DPSK)
    elif "dnn" in label_lower or "autoencoder" in label_lower:
        return 'b', 'd', '-.' # Blue Diamonds, Dash-Dot (DNN)
    else:
        return 'k', '.', ':'  # Black Dots, Dotted (Fallback)

def plot_throughput_kbps(ebn0_db, bler_tm, benchmarks_list, k_bits, n_symbols, symbol_rate_hz=1_000_000, save_path="throughput_kbps.png"):
    """
    Plots Effective Throughput with consistent coloring.
    """
    print(f"\n--- Calculating Throughput (Assuming {symbol_rate_hz/1e6} Msps) ---")
    
    # 1. Calculate Maximum Raw Data Rate for the TM System
    bits_per_symbol_tm = k_bits / n_symbols
    max_rate_kbps_tm = (bits_per_symbol_tm * symbol_rate_hz) / 1000.0
    
    print(f"Max Raw Rate (TM, K={k_bits}, N={n_symbols}): {max_rate_kbps_tm:.2f} kbps")

    plt.figure(figsize=(14, 7))

    # --- Max Rate Line ---
    plt.axhline(y=max_rate_kbps_tm, color='gray', linestyle=':', linewidth=2, 
                label=f'Max Theoretical Rate ({k_bits},{n_symbols})')

    # 2. Plot TM System
    label_tm = 'TM Autoencoder (7,4)'
    c, m, l = get_style_for_label(label_tm)
    throughput_tm = [max_rate_kbps_tm * (1 - b) for b in bler_tm]
    
    plt.plot(ebn0_db, throughput_tm, color=c, marker=m, linestyle=l, 
             label=label_tm, linewidth=2, markersize=8)

    highest_rate_found = max_rate_kbps_tm

    # 3. Benchmarks
    if benchmarks_list:
        iterable_benchmarks = benchmarks_list.values() if isinstance(benchmarks_list, dict) else benchmarks_list

        for benchmark in iterable_benchmarks:
            lbl = benchmark['label']
            label_lower = lbl.lower()
            if 'finite' in label_lower: continue 

            # --- DYNAMIC CONFIGURATION ---
            if 'pilot' in label_lower:
                # Hamming is usually K=4, but adds 1 pilot (N=8)
                curr_k = k_bits 
                curr_n = 8
            elif 'dpsk' in label_lower:
                # Standard DPSK: 8 bits data + 1 Ref = 9 Symbols
                curr_k = 8 
                curr_n = 9
            else:
                curr_k = k_bits
                curr_n = n_symbols

            # Calculate Max Rate
            bench_bits_per_symbol = curr_k / curr_n
            bench_max_rate = (bench_bits_per_symbol * symbol_rate_hz) / 1000.0
            
            if bench_max_rate > highest_rate_found:
                highest_rate_found = bench_max_rate

            tp_bench = [bench_max_rate * (1 - b) for b in benchmark['bler']]
            
            # Use consistent style
            c, m, l = get_style_for_label(lbl)
            plt.plot(benchmark['snr'], tp_bench, color=c, marker=m, linestyle=l, 
                     label=lbl, markersize=8)

    # --- Formatting ---
    plt.xlabel('$E_b/N_0$ (dB)', fontsize=14)
    plt.ylabel('Effective Throughput (kbps)', fontsize=14)
    plt.title(f'Throughput vs. $E_b/N_0$\n(Bandwidth: {symbol_rate_hz/1e6} MHz) in Rayleigh fading channel', fontsize=16)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Scale Y
    plt.ylim(bottom=0, top=highest_rate_found * 1.1) 
    
    plt.savefig(save_path)
    print(f"Throughput plot saved to {save_path}")
# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == '__main__':



        
    dec_clauses = 40000

    dec_T_factor = 0.16341435516042765
    
    dec_s = 8.988447894622514
    dec_max_included_literals = 20

    enc_clauses = 4900

    enc_T_factor = 0.387134110249462

    enc_s = 9.359579047718057

    param_clauses = 4400
    param_T_factor = 0.2729186626252808

    param_s = 9.088198002179771

    param_max_included_literals = 10
    enc_max_included_literals = 10

    dec_clause_drop_p = 0.16518591937933247
    dec_literal_drop_p = 0.26914363815495135






    # --- 1. Initialize TM Encoder and Decoder ---
    Normal_tm_decoder = TMCoalescedClassifier(
        number_of_clauses=dec_clauses, 
        T=int(dec_T_factor * dec_clauses), 
        s=dec_s, 
        max_included_literals=dec_max_included_literals, 
        platform="CPU",
        weighted_clauses=False, 
        type_iii_feedback=False,
        clause_drop_p=dec_clause_drop_p,
       # literal_drop_p=dec_literal_drop_p
    )

    Normal_tm_encoders = []
    for i in range(N_NORMAL*2):
        tm_enc = TMRegressor(
            enc_clauses, 
            int(enc_T_factor * enc_clauses), 
            enc_s, 
            platform="CPU",
            max_included_literals=enc_max_included_literals, 
            weighted_clauses=False,
        )
        Normal_tm_encoders.append(tm_enc)



    # param_estimatior

    param_estimator = []
    for i in range(2):
        tm_enc = TMRegressor(
            param_clauses, 
            int(param_T_factor * param_clauses), 
            param_s, 
            platform="CPU",
            max_included_literals=param_max_included_literals, 
            weighted_clauses=False,

        )
        param_estimator.append(tm_enc)
    TRAIN = True
    if TRAIN:
        # --- 2. Train the TM system ---
        Normal_tm_decoder, Normal_tm_encoders, param_estimator= train_autoencoder(Normal_tm_decoder,Normal_tm_encoders, param_estimator,
                                    epochs=EPOCHS_AE,
                                    snr_db=AE_TRAIN_SNR_DB,
                                    channel_type=CHANNEL_TYPE)
    else:
        # load trained models
        Normal_tm_decoder = joblib.load('models/best_tm_decoder.joblib')
        Normal_tm_encoders = []
        for i in range(2*N_NORMAL):
            tm_enc = joblib.load( f'models/best_tm_encoder_{i}.joblib')
            Normal_tm_encoders.append(tm_enc)
        
        param_estimator = []
        for i in range(2):
            tm_enc = joblib.load(f'models/best_param_est_{i}.joblib')
            param_estimator.append(tm_enc)
        



    # --- 3. Run Benchmarks ---
    bench_snr_range = np.arange(SNR_MIN, SNR_MAX + 1, 1)
    bench_results = get_classical_benchmarks(
        bench_snr_range, 
        batch_size=B_SIZE, 
        channel_type=CHANNEL_TYPE, 
        device=DEVICE
    )



    # Package the benchmark data into a list
    benchmarks_list = [
  
       bench_results['pilot_hamming'],
        bench_results['dpsk']
             
    ]

    # --- 4. Evaluate the TM System and Plot Comparison ---
    evaluate_and_plot_bler_output = evaluate_and_plot_bler(
        encoder=Normal_tm_encoders,
        decoder_tm=Normal_tm_decoder,
        Normal_param_estimator=param_estimator,
        channel_type=CHANNEL_TYPE,
        n=N_NORMAL,
        m=M_NORMAL,
        snr_min_db=SNR_MIN,
        snr_max_db=SNR_MAX,
        n_test_batches=10,
        batch_size=1000,
        benchmark_data_list=benchmarks_list, # <-- Pass list here
        save_path="bler_TM_vs_Benchmarks_R.png" # New save path
    )

    plot_throughput_kbps(
    ebn0_db=np.arange(SNR_MIN, SNR_MAX + 1, 1), # Your SNR axis
    bler_tm=evaluate_and_plot_bler_output[2],  # You need to capture the return value from your evaluation function
    benchmarks_list=benchmarks_list,
    k_bits=K_NORMAL,       
    n_symbols=N_NORMAL,    
    symbol_rate_hz=2_000_000, 
    save_path="throughput_kbps_comparison_R.png" 
    )
