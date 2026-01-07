import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.optimize import least_squares
import time

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
    """Create random masks/patterns."""
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    return ref

def reconstruct_cov(obj, ref):
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

def reconstruct_ls(obj, ref):
    """Reconstruct object using least squares optimization."""
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)
    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]

    # Flatten object
    obj_vec = obj_cropped.flatten()
    pixels = obj_vec.size

    # Build measurement matrix A: (N, pixels)
    A = ref_cropped.reshape(pixels, N).T  # shape (N, pixels)

    # Measurements
    b = A @ obj_vec  # size (N,)

    # Residuals function
    def residuals(x):
        return A @ x - b

    # Initial guess
    x0 = np.zeros(pixels)

    # Solve least squares
    res = least_squares(residuals, x0, method='trf')

    rec = res.x.reshape(h_min, w_min)
    return rec

# -----------------------------
# Experiment settings
# -----------------------------

object_sizes = [(32, 32)]
N_values = [50, 200, 500]
scale = 2

# -----------------------------
# Run experiment and compare
# -----------------------------

rows = len(object_sizes)
cols = len(N_values)

fig, axes = plt.subplots(rows*2, cols, figsize=(4*cols, 8*rows))  # 2 rows: cov + LS

times_cov = []
times_ls = []

for row_idx, size in enumerate(object_sizes):
    obj = create_object(size)
    for col_idx, N in enumerate(N_values):
        ref = create_patterns(size, N, scale=scale)

        # Covariance reconstruction
        start = time.time()
        rec_cov = reconstruct_cov(obj, ref)
        t_cov = time.time() - start
        times_cov.append((N, t_cov))

        ax = axes[0, col_idx] if rows == 1 else axes[2*row_idx, col_idx]
        ax.imshow(rec_cov, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Cov: Size={size}, N={N}\nTime={t_cov:.3f}s")

        # Least Squares reconstruction
        start = time.time()
        rec_ls = reconstruct_ls(obj, ref)
        t_ls = time.time() - start
        times_ls.append((N, t_ls))

        ax = axes[1, col_idx] if rows == 1 else axes[2*row_idx+1, col_idx]
        ax.imshow(rec_ls, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"LS: Size={size}, N={N}\nTime={t_ls:.3f}s")

plt.tight_layout()
plt.savefig("reconstruction_comparison.png", dpi=300)
plt.close()

# -----------------------------
# Print timing results
# -----------------------------
print("Timing results:")
print("N\tCov (s)\tLS (s)")
for (N, t_cov), (_, t_ls) in zip(times_cov, times_ls):
    print(f"{N}\t{t_cov:.4f}\t{t_ls:.4f}")

# -----------------------------
# Plot runtime comparison
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot([N for N,_ in times_cov], [t for _,t in times_cov], 'o-', label='Covariance')
plt.plot([N for N,_ in times_ls], [t for _,t in times_ls], 's-', label='Least Squares')
plt.xlabel("Number of patterns (N)")
plt.ylabel("Runtime (s)")
plt.title("Runtime Comparison: Covariance vs Least Squares")
plt.legend()
plt.grid(True)
plt.savefig("runtime_comparison.png", dpi=300)
plt.close()

print("Saved reconstruction_comparison.png and runtime_comparison.png")
