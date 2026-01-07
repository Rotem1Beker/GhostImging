import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# -----------------------------
# Functions
# -----------------------------

def create_object(shape, form="square"):
    """Create a synthetic object with given shape and form."""
    obj = np.zeros(shape)
    h, w = shape
    if form == "square":
        obj[h//4:3*h//4, w//4:3*w//4] = 1.0
    elif form == "circle":
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - h//2)**2 + (xx - w//2)**2 <= (min(h, w)//4)**2
        obj[mask] = 1.0
    return obj

def create_patterns(shape, N, scale=1):
    """Create random masks/patterns with controlled feature size."""
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    return ref

def reconstruct(obj, ref):
    """Reconstruct object using covariance formula."""
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

object_sizes = [(64, 64), (128, 128), (256, 256)]  # Different object dimensions
object_forms = ["square", "circle"]                 # Different shapes
N = 5000
scale = 2

# -----------------------------
# Run experiment
# -----------------------------

rows = len(object_forms)
cols = len(object_sizes)

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

for row_idx, form in enumerate(object_forms):
    for col_idx, size in enumerate(object_sizes):
        obj = create_object(size, form=form)
        ref = create_patterns(size, N, scale=scale)
        rec = reconstruct(obj, ref)
        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
        ax.imshow(rec, cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Form={form}, Size={size}")

plt.tight_layout()
plt.savefig("reconstruction_varying_object_properties.png", dpi=300)
plt.close()
print("Saved reconstruction_varying_object_properties.png")
