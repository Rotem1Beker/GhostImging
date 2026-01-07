import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time
import pylops
import pylops.optimization.sparsity as opt

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

def reconstruct_tv(obj, ref, mu=0.01, lamda=0.3, niter_out=50, niter_in=3):
    """Reconstruct object using Total Variation regularization (Split Bregman)."""
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)
    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]

    # Flatten object and measurement matrix
    pixels = h_min * w_min
    A = ref_cropped.reshape(pixels, N).T  # shape (N, pixels)
    b = A @ obj_cropped.flatten()          # measurements

    # המרת A ל-Linear Operator של PyLops
    Aop = pylops.MatrixMult(A)

    # Define TV operator
    D = pylops.FirstDerivative(pixels, kind='backward', edge=True)

    # Solve with Split Bregman TV
    xinv = opt.splitbregman(Aop, b, [D],
                            mu=mu,
                            epsRL1s=[lamda],
                            niter_outer=niter_out,
                            niter_inner=niter_in,
                            tol=1e-4,
                            tau=1.0,
                            iter_lim=30,
                            damp=1e-10)[0]

    return xinv.reshape(h_min, w_min)

# -----------------------------
# Experiment settings
# -----------------------------

object_sizes = [(16, 16)]
N_values = [50, 200, 500]
scale = 2

# -----------------------------
# Run experiment and compare
# -----------------------------

rows = len(object_sizes)
cols = len(N_values)

fig, axes = plt.subplots(rows*2, cols, figsize=(4*cols, 8*rows))  # 2 rows: cov + TV

times_cov = []
times_tv = []

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

        # TV reconstruction
        start = time.time()
        rec_tv = reconstruct_tv(obj, ref)
        t_tv = time.time() - start
        times_tv.append((N, t_tv))

        ax = axes[1, col_idx] if rows == 1 else axes[2*row_idx+1, col_idx]
        ax.imshow(rec_tv, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"TV: Size={size}, N={N}\nTime={t_tv:.3f}s")

plt.tight_layout()
plt.savefig("reconstruction_comparison_tv.png", dpi=300)
plt.close()

# Print timing results
print("Timing results:")
print("N\tCov (s)\tTV (s)")
for (N, t_cov), (_, t_tv) in zip(times_cov, times_tv):
    print(f"{N}\t{t_cov:.4f}\t{t_tv:.4f}")

# Plot runtime comparison
plt.figure(figsize=(6,4))
plt.plot([N for N,_ in times_cov], [t for _,t in times_cov], 'o-', label='Covariance')
plt.plot([N for N,_ in times_tv], [t for _,t in times_tv], 's-', label='TV Regularization')
plt.xlabel("Number of patterns (N)")
plt.ylabel("Runtime (s)")
plt.title("Runtime Comparison: Covariance vs TV")
plt.legend()
plt.grid(True)
plt.savefig("runtime_comparison_tv.png", dpi=300)
plt.close()

print("Saved reconstruction_comparison_tv.png and runtime_comparison_tv.png")
