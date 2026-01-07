import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# -----------------------------
# Functions
# -----------------------------

def create_object(shape):
    """Create a simple synthetic object: a white square in the center."""
    obj = np.zeros(shape)
    h, w = shape
    obj[h//4:3*h//4, w//4:3*w//4] = 1.0
    return obj

def create_patterns(shape, N, scale=1):
    """Create random masks/patterns with specified feature size (scale)."""
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    return ref

def reconstruct(obj, ref):
    """Reconstruct object using covariance formula (no noise)."""
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)
    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]
    test = np.zeros(N)
    for i in range(N):
        test[i] = np.sum(ref_cropped[:, :, i] * obj_cropped)
    rec = np.mean(ref_cropped * test[np.newaxis, np.newaxis, :], axis=2) - \
          np.mean(ref_cropped, axis=2) * np.mean(test)
    return rec

# -----------------------------
# Experiment settings
# -----------------------------

object_size = (128, 128)  # Fixed object size
N = 500  # Fixed number of patterns
scales = [1, 2, 4, 8, 16]  # Different feature sizes in the patterns

# -----------------------------
# Run experiment and save results
# -----------------------------

cols = len(scales)
fig, axes = plt.subplots(1, cols, figsize=(4*cols, 4))

for col_idx, scale in enumerate(scales):
    obj = create_object(object_size)
    ref = create_patterns(object_size, N, scale=scale)
    rec = reconstruct(obj, ref)
    ax = axes[col_idx] if cols > 1 else axes
    ax.imshow(rec, cmap='viridis')
    ax.axis('off')
    ax.set_title(f"Scale={scale}")

plt.tight_layout()
plt.savefig("reconstruction_varying_scale.png", dpi=300)
plt.close()
print("Saved reconstruction_varying_scale.png")
