"""
WESAD Multi-Modal Stress Detection with Leave-One-Subject-Out Cross-Validation
WITH PER-SUBJECT NORMALIZATION

UPDATE: tried adding chest ECG/EMG (RespiBAN) derived features, then backed
it out. The device is a forearm sleeve -- fine for EMG (that's the standard
placement, same idea as gesture armbands), but two electrodes on the same
forearm can't get a real ECG lead (needs two points spanning the torso, e.g.
wrist-to-wrist or wrist-to-ankle; same-limb picks up almost none of the
cardiac vector and gets swamped by local EMG). WESAD also has no wrist/
forearm EMG channel at all -- only chest (trapezius) EMG, a different muscle
group -- so there's nothing in this dataset to train a forearm-EMG signal
against either. Staying at 5 channels (ACC x3, EDA, TEMP) until there's a
sensor placement that actually has a WESAD-side match.

MODEL SWAP: BiLSTMAttention replaced with CNNGRUAttention (causal 1D convs
feeding a unidirectional GRU). A BiLSTM's backward pass needs future
timesteps (t5 to help classify t2) -- fine for offline analysis of a
complete pre-recorded window, but a live wearable predicting "cramp
incoming" at t2 doesn't have t5 yet. GRU is forward-only by construction;
the CNN front end is made causal too (left-padded only) so nothing in the
model ever depends on a future sample. Also lighter to run on embedded
hardware (GRU: fewer parameters than LSTM for the same hidden size; no
second backward pass to compute). Same constructor signature and forward()
contract as before, so train_one_fold/train_epoch/evaluate are unchanged.

CORRECTION (found while looking into this): the class remap in __main__
sorts raw WESAD labels {1,2,3,4} -> {0,1,2,3}, so remapped class 1 is actual
STRESS (label 2), not remapped class 2. Remapped class 2 is AMUSEMENT
(label 3). The FocalLoss comment below calling "class 2" stress predates
this fix and is wrong -- stress (class 1) was already the best-performing
class in the last run (0.69 precision / 0.65 recall); amusement (class 2) is
what's actually collapsing. Left the original comment below as history,
correcting here instead of rewriting it.
"""
import os
import pickle
import numpy as np
from collections import Counter
from datetime import datetime
import random
import json
import time

from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim


# CONFIG
# ------------------------------------------------------------

# Override with the WESAD_DATA_ROOT environment variable; falls back to the
# original development path when unset.
DATA_ROOT = os.environ.get(
    "WESAD_DATA_ROOT", "/Users/anitabhardwaj/AI_AGENT/POWERTHRU/WESAD/"
)
USE_SUBJECTS = None

"""New: once diagnose_subjects.py confirms a subject's raw data is broken
(dead sensor, huge gaps, mislabeled protocol), add it here so every run
stops burning a fold on it and dragging the mean/std down. Empty by
default -- fill in after looking at the plots.

ERROR: Subjects 14 and (maybe) 17 should be excluded based on diagnose_subjects.py
evidence, not guesswork:
    - Subject 14: eda_std = 0.025, roughly 15-35x lower than reference
    subjects (0.329 - 0.934) -- a near-flatlined EDA channel, consistent
    with a loose / disconnected electrode.
    - Subject 17: eda_stf = 0.248, second-lowest of all subjects checked
    plus an unusually narrow temperature range. Weaker evidence alone, 
    but comnined with catastrophic (-10%) accuracy across every loss
    function / oversampling combination tried so far, this is very unlikely
    to be the model's fault and may be physiological variablity that is unpredicatable
    and not viable for training."""
EXCLUDE_SUBJECTS = [14, 17]

# WESAD sampling rates
FS_LABEL = 700.0
FS_EDA   = 4.0
FS_TEMP  = 4.0
FS_ACC   = 32.0
FS_TARGET = 16.0

# Windowing
WINDOW_SEC = 60.0
STEP_SEC = 30.0
MIN_LABEL_PURITY = 0.8

SEED = 42
VALID_ANALYSIS_LABELS = {1,2,3,4}
MAX_EPOCHS = 60
# FIX: with only ~11-12 training subjects per fold and VAL_FRACTION=0.15,
# the validation split is just 1-2 subjects -- a small, noisy sample to
# estimate val loss from epoch to epoch. The old PATIENCE=8 was tripping
# early stopping after only a handful of epochs (some folds stopped by
# epoch 11-16), likely before the model had a real chance to converge.
# Raised to give it more room to ride out noisy validation swings.
PATIENCE = 15
VAL_FRACTION = 0.15

"""NEW: minority-calss handling. Combining all three (class weights, focal
loss, oversampling) is aggressive -- if stress starts getting OVER-predicted 
instead of under-predicted, turn one of these off rathern than stacking all three
at once.
UPDATE: oversampling + focal loss + class weights together over-corrected --
class 0 recall dropped from 0.52 to 0.31 as those samples got pushed into
class 2. Oversampling off by default: focal loss + class weights alone is 
a gentler correction. Re-enable oversampling only if class 2 recall is 
still too low with this combination."""
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
USE_OVERSAMPLING = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

OUTPUT_DIR = os.environ.get("WESAD_OUTPUT_DIR", "./wesad_runs")
CACHE_PATH = os.path.join(OUTPUT_DIR, "windowed_dataset_cache.npz")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------
# DATA LOADING WITH PER-SUBJECT NORMALIZATION


def resample_to_target(x, fs_in, fs_out, axis=0):
    if fs_in == fs_out:
        return x
    T = x.shape[axis]
    duration = T / fs_in
    T_new = int(round(duration * fs_out))
    return resample(x, T_new, axis=axis)


def downsample_labels(labels, fs_in=FS_LABEL, fs_out=FS_TARGET):
    ratio = int(round(fs_in / fs_out))
    return labels[::ratio]
def _fit_mask_to_length(mask, target_len):
    """FIX: the chest device (labels) and wrist device (EDA/TEMP/ACC) in WESAD
    are seperate recordings that don't cover the exact same sample count, so a ratio-based downsample
    of the label stream can come out shorter OR longer than the signal it's meant to mask. NumPy boolean
    indexing requires an exact length match, so this pads with False (exlcuded from stats) or truncates as needed
    instead of assuming the mask is always the longer of the two."""
    if len(mask) < target_len:
        pad = np.zeros(target_len - len(mask), dtype=bool)
        return np.concatenate([mask,pad])
    return mask[:target_len]

def normalize_subject_data(eda, temp, acc, labels_for_stats = None):
    """
    Apply per-subject z-score normalization to reduce subject-specific variability.
    Uses robust statistics (median, IQR) to handle outliers.

    FIX: if labels_for_stats is provided (same length as eda/temp/acc, at the
    *original* sampling rate before resampling), only samples witha valid analysis label (1-4)
    are used to compute the median/IQR. Previously all smaples -- including transient/movement-artificat
    segments labeled 0 and post-study segments labeled 5-7 -- were included, which
    skewed the baseline stats subjects differed on.
    """
    if labels_for_stats is not None:
        ratio = int(round(FS_LABEL / FS_EDA))
        mask_4hz = np.isin(labels_for_stats[::ratio], list(VALID_ANALYSIS_LABELS))
        mask_4hz = _fit_mask_to_length(mask_4hz, len(eda))
        if mask_4hz.sum() < 10:
            mask_4hz = np.ones(len(eda), dtype=bool)
    else:
        mask_4hz = np.ones(len(eda), dtype=bool)
    
    eda_ref = eda[mask_4hz]
    temp_ref = temp[mask_4hz]
    # EDA normalization
    eda_median = np.median(eda_ref)
    eda_iqr = np.percentile(eda_ref, 75) - np.percentile(eda_ref, 25)
    eda_normalized = (eda - eda_median) / (eda_iqr + 1e-8)
    
    # TEMP normalization
    temp_median = np.median(temp_ref)
    temp_iqr = np.percentile(temp_ref, 75) - np.percentile(temp_ref, 25)
    temp_normalized = (temp - temp_median) / (temp_iqr + 1e-8)
    
    if labels_for_stats is not None:
        ratio_acc = int(round(FS_LABEL / FS_ACC))
        mask_acc = np.isin(labels_for_stats[::ratio_acc], list(VALID_ANALYSIS_LABELS))
        mask_acc = _fit_mask_to_length(mask_acc, acc.shape[0])

        if mask_acc.sum() < 10:
            mask_acc = np.ones(acc.shape[0], dtype=bool)
    else:
        mask_acc = np.ones(acc.shape[0], dtype=bool)

    # ACC normalization (per axis)
    acc_normalized = np.zeros_like(acc)
    acc_ref = acc[mask_acc]
    for i in range(acc.shape[1]):
        acc_median = np.median(acc_ref[:, i])
        acc_iqr = np.percentile(acc_ref[:, i], 75) - np.percentile(acc_ref[:, i], 25)
        acc_normalized[:, i] = (acc[:, i] - acc_median) / (acc_iqr + 1e-8)
    
    return eda_normalized, temp_normalized, acc_normalized


def load_wesad_subject(path):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    wrist = data["signal"]["wrist"]
    labels = data["label"].reshape(-1)
    eda  = wrist["EDA"].reshape(-1)
    temp = wrist["TEMP"].reshape(-1)
    acc  = wrist["ACC"]
    # NOTE: BVP intentionally not loaded here -- PowerThru's wearable doesn't
    # have a PPG/BVP sensor, so training on it would produce a model that
    # relies on a signal the real device can never provide.
    #
    # NOTE: chest ECG/EMG also intentionally not loaded. The device is a
    # forearm sleeve, and a same-limb ECG lead isn't physically viable
    # (see module docstring), and WESAD has no wrist/forearm EMG channel to
    # train a forearm-placed EMG signal against -- chest EMG is a different
    # muscle group (trapezius) and wouldn't transfer.
    return eda, temp, acc, labels


def align_modalities(eda, temp, acc, labels):
    eda_r  = resample_to_target(eda,  FS_EDA,  FS_TARGET)
    temp_r = resample_to_target(temp, FS_TEMP, FS_TARGET)
    acc_r  = resample_to_target(acc,  FS_ACC,  FS_TARGET, axis=0)
    labels_r = downsample_labels(labels, FS_LABEL, FS_TARGET)
    
    T = min(len(eda_r), len(temp_r), len(labels_r), acc_r.shape[0])
    return eda_r[:T], temp_r[:T], acc_r[:T], labels_r[:T]


def make_windows(eda, temp, acc, labels, fs_target=FS_TARGET,
                 window_sec=WINDOW_SEC, step_sec=STEP_SEC, 
                 min_label_purity=MIN_LABEL_PURITY):
    win_len = int(window_sec * fs_target)
    step = int(step_sec * fs_target)
    
    X_list, y_list = [], []
    T = len(labels)
    
    for start in range(0, T - win_len + 1, step):
        end = start + win_len
        win_labels = labels[start:end]
        
        label_counts = Counter(win_labels)
        lbl, count = label_counts.most_common(1)[0]
        purity = count / len(win_labels)
        
        if lbl not in VALID_ANALYSIS_LABELS or purity < min_label_purity:
            continue
        
        acc_win  = acc[start:end, :]
        eda_win  = eda[start:end].reshape(-1, 1)
        temp_win = temp[start:end].reshape(-1, 1)
        seq = np.concatenate([acc_win, eda_win, temp_win], axis=1)
        
        X_list.append(seq)
        y_list.append(lbl)
    
    if not X_list:
        return np.empty((0, win_len, 5)), np.empty((0,), dtype=int)
    
    return np.stack(X_list, axis=0), np.array(y_list, dtype=int)


def load_all_subjects(data_root=DATA_ROOT, use_subjects=USE_SUBJECTS):
    X_all, y_all, subj_ids = [], [], []
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    
    subject_dirs = sorted([d for d in os.listdir(data_root) 
                          if os.path.isdir(os.path.join(data_root, d)) and d.startswith("S")])
    
    if not subject_dirs:
        raise FileNotFoundError(f"No S* subject directories found in {data_root}")
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    skipped = []
    for subj_dir in subject_dirs:
        try:
            sid = int(subj_dir[1:])
        except ValueError:
            continue
        
        if use_subjects is not None and sid not in use_subjects:
            continue

        if sid in EXCLUDE_SUBJECTS:
            print(f"Subject {sid}: excluded via EXCLUDE_SUBJECTS config")
            continue

        pkl_path = os.path.join(data_root, subj_dir, f"{subj_dir}.pkl")
        if not os.path.exists(pkl_path):
            continue
        
        try:
            eda, temp, acc, labels = load_wesad_subject(pkl_path)
            
            # Apply per-subject normalization BEFORE alignment and windowing
            #Fix: pass labels so normalization stats exlude transient segments;
            #note labels here are still at 700hz (pre-resample)
            eda_norm, temp_norm, acc_norm = normalize_subject_data(eda, temp, acc, labels_for_stats = labels)
            
            eda_r, temp_r, acc_r, labels_r = align_modalities(eda_norm, temp_norm, acc_norm, labels)
            X, y = make_windows(eda_r, temp_r, acc_r, labels_r)
            
            if len(X) > 0:
                print(f"Subject {sid}: {len(X)} windows, labels: {dict(Counter(y))}")
                X_all.append(X)
                y_all.append(y)
                subj_ids.append(np.full(len(y), sid))
        except (pickle.UnpicklingError, EOFError) as e:
            #FIX: specific, clear message from corrupted / truncated files
            # instead of a bare generic ERROR loading subject that 
            #looks like a bug
            print(f"SKIPPING Subject {sid}: file appears corrupted / truncated({e}). "
                  f"Re-download {pkl_path} and re-run")
            skipped.append(sid)
            continue
        except Exception as e:
            print(f"ERROR loading subject {sid}: {e}")
            skipped.append(sid)
            continue
    
    if not X_all:
        raise RuntimeError("No valid subjects found")
    
    if skipped:
        print(F"\nNote: {len(skipped)} subject(s) skipped due to load errors {skipped} "
            "Results below reflect only the successfully loaded subjects.\n")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    subj_ids = np.concatenate(subj_ids, axis=0)
    
    return X_all, y_all, subj_ids

def save_dataset_cache(X, y, subj_ids, path = CACHE_PATH):
    """NEW: save the windowed dataset so you don't have to re-parse and
    re-sample all raw WESAD pickles (slow) every time you run this."""
    os.makedirs(os.path.dirname(path), exist_ok = True)
    np.savez_compressed(path, X=X, y=y, subj_ids=subj_ids)
    print(f"Saved windowed dataset cache to {path}")

def load_dataset_cache(path=CACHE_PATH):
    """NEW: load a previously cached windowed dataset, if one exists."""
    if not os.path.exists(path):
        return None
    data = np.load(path)
    print(f"Loaded windowed dataset cache from {path}")
    return data["X"], data["y"], data["subj_ids"]

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

class WESADDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CausalConv1d(nn.Module):
    """1D convolution padded only on the left (past) side, so an output at
    time t depends only on inputs at time <= t -- no future leakage. A
    normal nn.Conv1d with symmetric padding lets each output peek a few
    samples into the future (half the kernel width); that's invisible on a
    pre-recorded window but not something a live wearable actually has."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=0, dilation=dilation)

    def forward(self, x):
        x = nn.functional.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CNNGRUAttention(nn.Module):
    """
    Drop-in replacement for the old BiLSTMAttention: same constructor
    signature (input_size, hidden_size, num_layers, num_classes, dropout)
    and same forward(x) -> logits contract, so train_one_fold / train_epoch
    / evaluate need no changes beyond the class name at instantiation.

    CausalConv1d layers pick up local bursts/spikes per channel (EMG
    activation, EDA rises, etc.), then a unidirectional GRU tracks how those
    features evolve over the window -- forward-only, matching how the
    wearable actually receives samples in real time. Attention pooling is
    kept from the original architecture; it's still causal here since it
    only pools over GRU outputs already computed from current-and-past
    samples, never a future one.
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=2,
                 num_classes=4, dropout=0.4, cnn_channels=(32, 64), kernel_size=5):
        super().__init__()
        conv_layers = []
        in_ch = input_size
        for out_ch in cnn_channels:
            conv_layers += [
                CausalConv1d(in_ch, out_ch, kernel_size),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*conv_layers)

        # NEW: unidirectional (bidirectional=False) -- this is the actual
        # real-time fix. GRU chosen over LSTM per the ranking this was built
        # from: fewer parameters for the same hidden size, comparable
        # accuracy, easier to run on an embedded target like an RP2350.
        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False, # enable if your want a BiCNN + GRU but probably harder to run
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_fc = nn.Linear(hidden_size, 1)
        self.fc1 = nn.Linear(hidden_size, 48)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(48, num_classes)

    def forward(self, x):
        # x arrives as (batch, time, channels); conv1d wants
        # (batch, channels, time).
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # back to (batch, time, channels) for the GRU

        gru_out, _ = self.gru(x)
        attn_scores = self.attn_fc(torch.tanh(gru_out)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (gru_out * attn_weights).sum(dim=1)

        out = torch.relu(self.fc1(context))
        out = self.dropout(out)
        return self.fc2(out)

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(dataloader), 100.0 * correct / total, all_preds, all_labels

def split_train_val_by_subject(X_train, y_train, train_subj_ids, val_fraction, seed):
    """
    FIX: early stopping was validating on a random 15% of *windows*, which could
    include windows from the same subjects used for training -- that measures
    within-subject fit, not cross-subject generalization, which is the whole point 
    of LOSO. This holds out entire subjects instead, so validation loss actually reflects
    "does this generalize to people the model hasn't seen," same as the real test fold does.
    """
    unique_train_subjects = np.unique(train_subj_ids)
    n_val_subjects = max(1, int(round(len(unique_train_subjects) * val_fraction)))
    rng = np.random.RandomState(seed)
    val_subjects = rng.choice(unique_train_subjects, size=n_val_subjects, replace=False)

    val_mask = np.isin(train_subj_ids, val_subjects)
    tr_mask = ~val_mask
    print(f"    Validation subjects (held out from inner training): {sorted(val_subjects.tolist())}")

    return X_train[tr_mask], X_train[val_mask], y_train[tr_mask], y_train[val_mask]

def safe_class_weights(y_train, num_classes, device):
    """
    FIX: build a weight vector covering the FULL 0..num_classes-1 range.
    The original version computed weights only from classes present in 
    y_train via np.unique, so if a fold's training set (rare, but possible)
    is missing a class, the resulting weight tensor length would not match
    num_classes and CrossEntropyLoss would error or silently misalign weights
    to the wrong class indices. """
    counts = np.array([np.sum(y_train == c) for c in range(num_classes)], dtype = np.float64)

    counts = np.maximum(counts, 1.0)
    weights = len(y_train) / (num_classes * counts)
    return torch.FloatTensor(weights).to(device)

class FocalLoss(nn.Module):
    """
    NEW: standard weighted CE was still leaving class 2 (stress, ~155 samples total)
    with 0.26 precision / 0.37 recal. Focal loss down-weights
    the *easy* majority-class examples the model already classifies confidently, so 
    gradient signal concentrates on the harder, rarer, exmaples instead of being dominated
    by volume of class 0/majority. Combine with class weights (alpha) for both effects together.

    CORRECTION: "class 2" above is wrong -- after the label remap in
    __main__, class 2 is amusement (label 3), not stress (which is class 1,
    label 2). See the module-level docstring UPDATE note. Left the original
    text as-is above for history.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma

        return (focal_term * ce_loss).mean()

def train_one_fold(X_train, y_train, train_subj_ids, X_test, y_test, num_classes,
                   input_channels, device, max_epochs = MAX_EPOCHS, patience=PATIENCE,
                   val_fraction=VAL_FRACTION, model_save_path = None, val_seed = SEED):
    """Train model for one LOSO fold with early stopping on an internal validation split
    carved out of teh training subjects (never from the held-out test subjects.)"""
    # FIX: this previously called sklearn's train_test_split, which splits by
    # random *window*, not by subject. That let windows from the same subject
    # appear in both the "train" and "validation" halves within a fold, so
    # early stopping was measuring within-subject fit instead of cross-subject
    # generalization -- the split_train_val_by_subject() helper that was meant
    # to fix this was defined but never actually called. Wiring it in here.
    X_tr, X_val, y_tr, y_val = split_train_val_by_subject(
        X_train, y_train, train_subj_ids, val_fraction=val_fraction, seed=val_seed
    )
    # Scale features using StandardScaler (data already normalized per-subject)
    scaler = StandardScaler()
    X_tr_flat = X_tr.reshape(-1, input_channels)
    X_tr_scaled = scaler.fit_transform(X_tr_flat).reshape(X_tr.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, input_channels)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, input_channels)).reshape(X_test.shape)

    # Create datasets
    train_dataset = WESADDataset(X_tr_scaled, y_tr)
    val_dataset = WESADDataset(X_val_scaled, y_val)
    test_dataset = WESADDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if USE_OVERSAMPLING:
        class_counts_tr = np.array([max(np.sum(y_tr == c), 1) for c in range(num_classes)], dtype=np.float64)
        sample_weights = np.array([1.0 / class_counts_tr[label] for label in y_tr])
        sampler = WeightedRandomSampler(
            weights = sample_weights, num_samples = len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size = 32, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Model
    model = CNNGRUAttention(
        input_size=input_channels,
        hidden_size=64,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.4
    ).to(device)
    
    # Class weights
    class_weights = safe_class_weights(y_tr, num_classes, device)
    
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha = class_weights, gamma=FOCAL_GAMMA)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    # NEW: with PATIENCE raised to 15, folds now train for more epochs before
    # stopping. A fixed LR for that whole stretch tends to plateau or start
    # oscillating once the model is near a local minimum. Decaying LR when
    # val loss stalls lets later epochs make finer progress instead of just
    # circling the same loss value until patience runs out.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    # Training loop (simplified - no validation, just train for fixed epochs)
    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch [{epoch+1}/{max_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k , v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"    Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break
    # Final evaluation on test subject
    #FIX: restore the best checkpoint (lowest val loss) rather than
    #evaluating whatever the model looked like after a fixed 30 epochs.
    if best_state is not None:
        model.load_state_dict(best_state)

    if model_save_path is not None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale":scaler.scale_,
            "num_classes":num_classes,
            "input_channels":input_channels,
        }, model_save_path)
        print(f"    Saved model checkpoint to {model_save_path}")
        
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    return test_acc, test_preds, test_labels


# ------------------------------------------------------------
# LOSO CROSS-VALIDATION
# ------------------------------------------------------------

def loso_cross_validation(X, y, subj_ids, num_classes, input_channels, device, output_dir = OUTPUT_DIR):
    """
    Leave-One-Subject-Out Cross-Validation
    """
    unique_subjects = np.unique(subj_ids)
    print(f"\n{'='*60}")
    print(f"Leave-One-Subject-Out Cross-Validation")
    print(f"{'='*60}")
    print(f"Total subjects: {len(unique_subjects)}")
    print(f"Subjects: {sorted(unique_subjects)}\n")
    
    all_test_accs = []
    all_test_f1s_weighted = []
    all_test_f1s_macro = []
    all_preds = []
    all_labels = []
    all_fold_subj_ids = []
    
    for i, test_subj in enumerate(unique_subjects, 1):
        print(f"[Fold {i}/{len(unique_subjects)}] Test subject: {test_subj}")
        
        # Split data
        test_mask = (subj_ids == test_subj)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        train_subj_ids = subj_ids[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"  Train: {len(X_train)} windows, Test: {len(X_test)} windows")
        print(f"  Train labels: {dict(Counter(y_train))}")
        print(f"  Test labels: {dict(Counter(y_test))}")
        """New: each fold's model is saved as its own checkpoint, named by
        the subject that was held out for testing."""
        fold_checkpoint_path = os.path.join(output_dir, "models", f"fold_test_subj_{test_subj}.pt")
        # Train and evaluate
        # FIX: split_train_val_by_subject was always called with the same
        # fixed SEED, so across folds it kept landing on nearly the same 1-2
        # validation subjects (e.g. subject 2 showed up in almost every fold's
        # validation set). Varying the seed per fold gives each fold a
        # different, still-reproducible validation subject sample.
        test_acc, test_preds, test_labels = train_one_fold(
            X_train, y_train, train_subj_ids, X_test, y_test,
            num_classes, input_channels, device,
            model_save_path = fold_checkpoint_path,
            val_seed = SEED + i
        )
        
        # Calculate F1 score
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
        test_f1_macro = f1_score(test_labels, test_preds, average='macro')
        
        all_test_accs.append(test_acc)
        all_test_f1s_weighted.append(test_f1_weighted)
        all_test_f1s_macro.append(test_f1_macro)
        all_preds.extend(test_preds)
        all_labels.extend(test_labels)
        # NEW: tag every prediction with the subject it came from (the test
        # subject for this fold) so later diagnostics can check whether
        # confused windows cluster onto specific subjects instead of guessing
        # from plots alone.
        all_fold_subj_ids.extend([test_subj] * len(test_labels))
        
        print(f"  → Test Accuracy: {test_acc:.2f}%, F1: {test_f1_macro:.4f}\n")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"LOSO Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {np.mean(all_test_accs):.2f}% ± {np.std(all_test_accs):.2f}%")
    print(f"Mean F1 (weighted): {np.mean(all_test_f1s_weighted):.4f} ± {np.std(all_test_f1s_weighted):.4f}")
    print(f"Mean F1 (macro): {np.mean(all_test_f1s_macro):.4f} +/- {np.std(all_test_f1s_macro):.4f}")
    print(f"\nPer-subject accuracies:")
    for subj, acc in zip(unique_subjects, all_test_accs):
        print(f"  Subject {subj}: {acc:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Overall Classification Report")
    print(f"{'='*60}")
    print(classification_report(all_labels, all_preds))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return {
        'mean_acc': np.mean(all_test_accs),
        'std_acc': np.std(all_test_accs),
        'mean_f1_weighted': np.mean(all_test_f1s_weighted),
        'mean_f1_macro': np.mean(all_test_f1s_macro),
        'per_subject_accs': dict(zip(unique_subjects, all_test_accs)),
        'all_preds': all_preds,
        'all_labels': all_labels,
        'all_subj_ids': all_fold_subj_ids
    }

def save_results(results, output_dir=OUTPUT_DIR):
    """NEW: write the summary metrics to a timestamped JSON file, and the
    raw per-window predictions/labels to a .npz, so a run's results
    survive after the terminal closes and can be compared accross runs."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": timestamp,
        "mean_acc": float(results["mean_acc"]),
        "std_acc": float(results["std_acc"]),
        "mean_f1_weighted": float(results["mean_f1_weighted"]),
        "mean_f1_macro": float(results["mean_f1_macro"]),
        "per_subject_accs": {str(k): float(v) for k, v in results["per_subject_accs"].items()},
    }
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent = 2)
    
    preds_path = os.path.join(output_dir, f"preds_labels_{timestamp}.npz")
    np.savez_compressed(
        preds_path,
        preds = np.array(results["all_preds"]),
        labels = np.array(results["all_labels"]),
        # NEW: subject ID for each prediction, in the same order as preds/labels
        # (the test subject for whichever fold produced that prediction).
        subj_ids = np.array(results["all_subj_ids"]),
    )
    
    print(f"Saved summary metrics to {json_path}")
    print(f"Saved raw predictions/label to {preds_path}")
    return json_path, preds_path


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    # FIX: set_seed() was defined (with a docstring claiming reproducibility
    # was already handled) but never actually called anywhere in the script.
    # Without this, torch's RNG state differs every run based on process
    # start entropy, so model init and DataLoader shuffling change from run
    # to run even with identical code and identical cached data -- which is
    # exactly why repeated runs were landing anywhere from 46.8% to 50.5% on
    # the same dataset. Calling it here makes runs reproducible so future
    # changes can actually be measured against a stable baseline.
    set_seed(SEED)

    print("="*60)
    print("WESAD LOSO Cross-Validation WITH SUBJECT NORMALIZATION")
    print("="*60)
    
    try:
        """NEW: reuse a cached windowed dataset if one exists, instead of
        re-parsing and re-sampling all raw WESAD pickles every run."""
        cached = load_dataset_cache()
        if cached is not None:
            X, y, subj_ids = cached
            # FIX: cache shape can go stale any time the windowing pipeline's
            # channel count changes (e.g. after adding/removing a signal).
            # Detect a mismatch against the current pipeline and force a
            # full reload rather than silently training on the wrong shape.
            expected_channels = 5
            if X.shape[2] != expected_channels:
                print(f"Cached dataset has {X.shape[2]} channels but pipeline now produces "
                      f"{expected_channels} -- discarding stale cache and reloading raw data.")
                cached = None
        if cached is None:
            X, y, subj_ids = load_all_subjects()
            save_dataset_cache(X, y, subj_ids)
        else:
            # FIX: a cached dataset may have been built before EXCLUDE_SUBJECTS
            # was last edited (e.g. subject 17 added after the cache already
            # existed). Loading straight from cache silently ignored that
            # change and kept feeding excluded subjects into every fold.
            # Filter here so config changes take effect without needing to
            # know to delete the cache file by hand.
            exclude_mask = np.isin(subj_ids, EXCLUDE_SUBJECTS)
            if exclude_mask.any():
                removed = sorted(set(subj_ids[exclude_mask].tolist()))
                print(f"Filtering excluded subjects {removed} out of cached dataset "
                      f"(present in EXCLUDE_SUBJECTS but cache predates that config)")
                keep_mask = ~exclude_mask
                X, y, subj_ids = X[keep_mask], y[keep_mask], subj_ids[keep_mask]
        
        print(f"\nTotal windows: {len(X)}")
        print(f"Window shape: {X.shape[1:]} (time_steps, channels)")
        print(f"Label distribution: {Counter(y)}")
        
        # Remap labels to 0-indexed
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_remapped = np.array([label_map[lbl] for lbl in y])
        
        print(f"\nLabel mapping: {label_map}")
        print("(raw WESAD labels: 1=baseline, 2=stress, 3=amusement, 4=meditation --")
        print(" so with this mapping remapped class 1 is STRESS, not remapped class 2.)")
        
        num_classes = len(unique_labels)
        input_channels = X.shape[2]
        
        # Run LOSO CV
        results = loso_cross_validation(
            X, y_remapped, subj_ids,
            num_classes, input_channels, DEVICE
        )
        
        print(f"\n{'='*60}")
        print(f"Final Result: {results['mean_acc']:.2f}% +/- {results['std_acc']:.2f}%")
        print(f"Mean F1 (macro, exposes minority-class performance): {results['mean_f1_macro']:.4f}")
        print(f"{'='*60}")

        save_results(results)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
