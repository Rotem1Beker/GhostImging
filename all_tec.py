import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.optimize import least_squares
import time
import pylops
import pylops.optimization.sparsity as opt

# -----------------------------
# Functions
# -----------------------------

def create_object(shape):
    obj = np.zeros(shape)
    h, w = shape
    obj[h//4:3*h//4, w//4:3*w//4] = 1.0
    return obj

def create_patterns(shape, N, scale=1):
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

def reconstruct_pinv(obj, ref):
    """Reconstruct object using pseudoinverse."""
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
    b = A @ obj_vec   # size (N,)

    # Solve with pseudoinverse
    A_pinv = np.linalg.pinv(A)  # shape (pixels, N)
    x_hat = A_pinv @ b          # shape (pixels,)

    rec = x_hat.reshape(h_min, w_min)
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

object_sizes = [(32, 32)]
N_values = [50, 200, 500]
scale = 2
methods = ['Cov', 'Pinv', 'LS', 'TV']

# -----------------------------
# Run experiment and compare
# -----------------------------

fig, axes = plt.subplots(len(object_sizes)*len(methods), len(N_values), figsize=(4*len(N_values), 4*len(object_sizes)*len(methods)))

timings = {m: [] for m in methods}

for row_idx, size in enumerate(object_sizes):
    obj = create_object(size)
    for col_idx, N in enumerate(N_values):
        ref = create_patterns(size, N, scale=scale)

        # Covariance
        start = time.time()
        rec_cov = reconstruct_cov(obj, ref)
        t_cov = time.time() - start
        timings['Cov'].append((N, t_cov))
        ax = axes[0 + row_idx*len(methods), col_idx]
        ax.imshow(rec_cov, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Cov: N={N}\n{t_cov:.3f}s")

        # Pseudoinverse
        start = time.time()
        rec_pinv = reconstruct_pinv(obj, ref)
        t_pinv = time.time() - start
        timings['Pinv'].append((N, t_pinv))
        ax = axes[1 + row_idx*len(methods), col_idx]
        ax.imshow(rec_pinv, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Pinv: N={N}\n{t_pinv:.3f}s")

        # Least Squares
        start = time.time()
        rec_ls = reconstruct_ls(obj, ref)
        t_ls = time.time() - start
        timings['LS'].append((N, t_ls))
        ax = axes[2 + row_idx*len(methods), col_idx]
        ax.imshow(rec_ls, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"LS: N={N}\n{t_ls:.3f}s")

        # TV
        start = time.time()
        rec_tv = reconstruct_tv(obj, ref)
        t_tv = time.time() - start
        timings['TV'].append((N, t_tv))
        ax = axes[3 + row_idx*len(methods), col_idx]
        ax.imshow(rec_tv, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"TV: N={N}\n{t_tv:.3f}s")

plt.tight_layout()
plt.savefig("reconstruction_comparison_all_methods.png", dpi=300)
plt.close()

# -----------------------------
# Print timing results
# -----------------------------
print("Timing results (s):")
print("N\tCov\tPinv\tLS\tTV")
for idx, N in enumerate(N_values):
    print(f"{N}\t{timings['Cov'][idx][1]:.4f}\t{timings['Pinv'][idx][1]:.4f}\t{timings['LS'][idx][1]:.4f}\t{timings['TV'][idx][1]:.4f}")

# Plot runtime comparison
plt.figure(figsize=(6,4))
for method in methods:
    plt.plot([N for N,_ in timings[method]], [t for _,t in timings[method]], marker='o', label=method)
plt.xlabel("Number of patterns (N)")
plt.ylabel("Runtime (s)")
plt.title("Runtime Comparison: All Methods")
plt.legend()
plt.grid(True)
plt.savefig("runtime_comparison_all_methods.png", dpi=300)
plt.close()

print("Saved reconstruction_comparison_all_methods.png and runtime_comparison_all_methods.png")
