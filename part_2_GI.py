#גרסה ראשונה, לא אהבתי כי כל הגרפים יוצאים בשורה אחת
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# פונקציה שמייצרת אובייקט מלאכותי עם שליטה בניגודיות וצורה
def create_object(shape, contrast=(0.0, 1.0), form="square"):
    obj = np.full(shape, contrast[0])
    h, w = shape
    if form == "square":
        obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    elif form == "circle":
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - h//2)**2 + (xx - w//2)**2 <= (min(h, w)//4)**2
        obj[mask] = contrast[1]
    return obj

# פונקציה שמייצרת דפוסי אור עם קונטרסט נשלט
def create_patterns(shape, N, scale=1, mask_contrast=1.0):
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    # שינוי קונטרסט
    ref = ref * mask_contrast
    return ref

# פונקציה שמוסיפה רעש (Shot + Thermal)
def add_noise(test, ref, obj, lambda_param=50, mu=0, sigma=1):
    noisy = np.zeros_like(test)
    for i in range(len(test)):
        # shot noise
        lam = lambda_param * np.sum(ref[:, :, i] * obj)
        shot = np.random.poisson(lam)
        # thermal noise
        thermal = np.random.normal(mu, sigma)
        noisy[i] = shot + thermal
    return noisy

# שחזור לפי נוסחת הקו-וריאנס
def reconstruct(obj, ref, lambda_param=50, mu=0, sigma=1):
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)

    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]

    # מדידות bucket עם רעש
    test = add_noise(np.zeros(N), ref_cropped, obj_cropped,
                     lambda_param=lambda_param, mu=mu, sigma=sigma)

    rec = np.mean(ref_cropped * test[np.newaxis, np.newaxis, :], axis=2) \
          - np.mean(ref_cropped, axis=2) * np.mean(test)
    return rec

# =====================
# ניסוי עם פרמטרים שונים
# =====================

object_sizes = [(100, 200)]
N_values = [500, 2000]
scales = [1, 5]
contrasts = [(0.0, 1.0), (0.1, 0.9)]
mask_contrasts = [0.5, 1.0, 2.0]

total = len(object_sizes) * len(N_values) * len(scales) * len(contrasts) * len(mask_contrasts)
fig, axes = plt.subplots(1, total, figsize=(3*total, 4))

if total == 1:
    axes = [axes]

idx = 0
for size in object_sizes:
    for N in N_values:
        for scale in scales:
            for contrast in contrasts:
                for mask_c in mask_contrasts:
                    obj = create_object(size, contrast=contrast, form="square")
                    ref = create_patterns(size, N, scale=scale, mask_contrast=mask_c)
                    rec = reconstruct(obj, ref,
                                      lambda_param=50, mu=0, sigma=5)  # שינוי רעש
                    ax = axes[idx]
                    ax.imshow(rec, cmap='viridis')
                    ax.axis('off')
                    ax.set_title(f"Size={size}\nN={N}, s={scale}, c={contrast}, mC={mask_c}")
                    idx += 1

plt.tight_layout()
plt.show()
"""

#קוד שני, לא אהבתי כי בקושי רואים את השינויים
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# פונקציה שמייצרת אובייקט מלאכותי עם שליטה בניגודיות וצורה
def create_object(shape, contrast=(0.0, 1.0), form="square"):
    obj = np.full(shape, contrast[0])
    h, w = shape
    if form == "square":
        obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    elif form == "circle":
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - h//2)**2 + (xx - w//2)**2 <= (min(h, w)//4)**2
        obj[mask] = contrast[1]
    return obj

# פונקציה שמייצרת דפוסי אור עם קונטרסט נשלט
def create_patterns(shape, N, scale=1, mask_contrast=1.0):
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    # שינוי קונטרסט
    ref = ref * mask_contrast
    return ref

# פונקציה שמוסיפה רעש (Shot + Thermal)
def add_noise(test, ref, obj, lambda_param=50, mu=0, sigma=1):
    noisy = np.zeros_like(test)
    for i in range(len(test)):
        # shot noise
        lam = lambda_param * np.sum(ref[:, :, i] * obj)
        shot = np.random.poisson(lam)
        # thermal noise
        thermal = np.random.normal(mu, sigma)
        noisy[i] = shot + thermal
    return noisy

# שחזור לפי נוסחת הקו-וריאנס
def reconstruct(obj, ref, lambda_param=50, mu=0, sigma=1):
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)

    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]

    # מדידות bucket עם רעש
    test = add_noise(np.zeros(N), ref_cropped, obj_cropped,
                     lambda_param=lambda_param, mu=mu, sigma=sigma)

    rec = np.mean(ref_cropped * test[np.newaxis, np.newaxis, :], axis=2) \
          - np.mean(ref_cropped, axis=2) * np.mean(test)
    return rec

# =====================
# ניסוי עם פרמטרים שונים
# =====================

object_sizes = [(100, 200)]
N_values = [500, 2000]
scales = [1, 5]
contrasts = [(0.0, 1.0), (0.1, 0.9)]
mask_contrasts = [0.5, 1.0, 2.0]

total = len(object_sizes) * len(N_values) * len(scales) * len(contrasts) * len(mask_contrasts)

# סידור של 3 גרפים בכל שורה
cols = 3
rows = int(np.ceil(total / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

# הפיכת axes למערך שטוח
axes = axes.flatten()

idx = 0
for size in object_sizes:
    for N in N_values:
        for scale in scales:
            for contrast in contrasts:
                for mask_c in mask_contrasts:
                    obj = create_object(size, contrast=contrast, form="square")
                    ref = create_patterns(size, N, scale=scale, mask_contrast=mask_c)
                    rec = reconstruct(obj, ref,
                                      lambda_param=50, mu=0, sigma=5)  # שינוי רעש
                    ax = axes[idx]
                    ax.imshow(rec, cmap='gray')  # שחור לבן
                    ax.axis('off')
                    ax.set_title(f"N={N}, s={scale}, c={contrast}, mC={mask_c}")
                    idx += 1

# הסתרת צירים ריקים אם יש יותר משבצות מתוצאות
for ax in axes[idx:]:
    ax.axis('off')

plt.tight_layout()
plt.savefig("reconstruction_results.png", dpi=300)  # שמירה לקובץ
plt.close()
"""

#זה בסדר אבל רק רעש גאוסייני
"""
import numpy as np
import matplotlib.pyplot as plt

# Function to create a simple object (red square in the center on black background)
def create_object(shape):
    obj = np.zeros((shape[0], shape[1], 3))
    h, w, _ = obj.shape
    obj[h//4:3*h//4, w//4:3*w//4, 0] = 1.0   # red square
    return obj

# Function to add Gaussian noise
def add_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 1)  # keep values in [0,1]
    return noisy

# Create object
obj = create_object((64, 64))

# Different noise levels
noise_levels = [0.05, 0.3, 0.7, 1.0]

# Plot images and pixel distributions
fig, axes = plt.subplots(len(noise_levels), 2, figsize=(10, 10))

for i, sigma in enumerate(noise_levels):
    noisy_img = add_noise(obj, sigma)
    
    # Show noisy image
    axes[i, 0].imshow(noisy_img)
    axes[i, 0].set_title(f"Image with noise σ={sigma}")
    axes[i, 0].axis('off')
    
    # Show histogram of pixel values
    axes[i, 1].hist(noisy_img.ravel(), bins=50, color='gray')
    axes[i, 1].set_title(f"Pixel value distribution (σ={sigma})")
    axes[i, 1].set_xlim(0, 1)

plt.tight_layout()
plt.show()
"""
#זה הוציא המון גרפים אבל רציתי בקובץ
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pandas as pd
import seaborn as sns

# -----------------------------
# Functions
# -----------------------------

def create_object(shape, contrast=(0.0, 1.0), form="square"):
    obj = np.full(shape, contrast[0])
    h, w = shape
    if form == "square":
        obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    elif form == "circle":
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - h//2)**2 + (xx - w//2)**2 <= (min(h, w)//4)**2
        obj[mask] = contrast[1]
    return obj

def create_patterns(shape, N, scale=1, mask_contrast=1.0):
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    ref = ref * mask_contrast
    return ref

def add_noise_buckets(ref, obj, lambda_param=50, mu=0, sigma=1):
    N = ref.shape[2]
    buckets = np.zeros(N)
    for i in range(N):
        lam = lambda_param * np.sum(ref[:, :, i] * obj)
        shot = np.random.poisson(lam)
        thermal = np.random.normal(mu, sigma)
        buckets[i] = shot + thermal
    return buckets

def reconstruct(obj, ref, lambda_param=50, mu=0, sigma=1):
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)
    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]
    buckets = add_noise_buckets(ref_cropped, obj_cropped,
                                lambda_param=lambda_param, mu=mu, sigma=sigma)
    rec = np.mean(ref_cropped * buckets[np.newaxis, np.newaxis, :], axis=2) \
          - np.mean(ref_cropped, axis=2) * np.mean(buckets)
    return rec

def mse(a, b):
    return np.mean((a - b)**2)

# -----------------------------
# Experiment settings
# -----------------------------

object_shapes = ["square", "circle"]
object_sizes = [(50, 50), (100, 100)]
object_contrasts = [(0.0, 1.0), (0.2, 0.8)]
mask_contrasts = [0.5, 1.0, 2.0]
noise_params = [
    {"lambda_param": 10, "mu": 0, "sigma": 1},
    {"lambda_param": 50, "mu": 0, "sigma": 5},
    {"lambda_param": 100, "mu": 0, "sigma": 20}
]
N = 200
scale = 2

# -----------------------------
# Run experiments
# -----------------------------
results = []

for form in object_shapes:
    for size in object_sizes:
        for contrast in object_contrasts:
            obj = create_object(size, contrast=contrast, form=form)
            for mask_c in mask_contrasts:
                ref = create_patterns(size, N, scale=scale, mask_contrast=mask_c)
                for noise in noise_params:
                    rec = reconstruct(obj, ref,
                                      lambda_param=noise["lambda_param"],
                                      mu=noise["mu"],
                                      sigma=noise["sigma"])
                    error = mse(obj, rec)
                    results.append({
                        "shape": form,
                        "size": size,
                        "contrast": contrast,
                        "mask_contrast": mask_c,
                        "lambda_param": noise["lambda_param"],
                        "sigma": noise["sigma"],
                        "mse": error,
                        "reconstruction": rec
                    })

# -----------------------------
# Visualization: reconstructions
# -----------------------------
cols = 3
rows = int(np.ceil(len(results) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()

for i, res in enumerate(results):
    ax = axes[i]
    ax.imshow(res["reconstruction"], cmap="gray")
    ax.axis("off")
    ax.set_title(f"{res['shape']}, size={res['size']}\n"
                 f"c={res['contrast']}, mC={res['mask_contrast']}\n"
                 f"λ={res['lambda_param']}, σ={res['sigma']}\nMSE={res['mse']:.2f}")

for ax in axes[len(results):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
plt.savefig("reconstruction_all_variations.png", dpi=300)

# -----------------------------
# Convert results to DataFrame
# -----------------------------
df = pd.DataFrame([{
    "shape": r["shape"],
    "size": r["size"],
    "contrast": r["contrast"],
    "mask_contrast": r["mask_contrast"],
    "lambda": r["lambda_param"],
    "sigma": r["sigma"],
    "mse": r["mse"]
} for r in results])

# -----------------------------
# Visualization: MSE trends
# -----------------------------
plt.figure(figsize=(12,4))
sns.scatterplot(x="lambda", y="mse", hue="sigma", style="shape", size="mask_contrast", data=df, palette="viridis", s=100)
plt.title("MSE vs Lambda (Shot Noise), colored by Sigma (Thermal Noise), mask contrast size-coded")
plt.xlabel("Lambda (Shot noise)")
plt.ylabel("MSE")
plt.show()

plt.figure(figsize=(12,4))
sns.scatterplot(x=df["contrast"].apply(lambda x: x[1]), y="mse", hue="shape", size="mask_contrast", data=df, palette="coolwarm", s=100)
plt.title("MSE vs Object Contrast (max pixel value)")
plt.xlabel("Object Contrast (max)")
plt.ylabel("MSE")
plt.show()

plt.figure(figsize=(12,4))
sns.scatterplot(x="mask_contrast", y="mse", hue="shape", style="contrast", size="lambda", data=df, palette="magma", s=100)
plt.title("MSE vs Mask Contrast")
plt.xlabel("Mask Contrast")
plt.ylabel("MSE")
plt.show()
"""

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pandas as pd
import seaborn as sns

# -----------------------------
# Functions
# -----------------------------

def create_object(shape, contrast=(0.0, 1.0), form="square"):
    obj = np.full(shape, contrast[0])
    h, w = shape
    if form == "square":
        obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    elif form == "circle":
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - h//2)**2 + (xx - w//2)**2 <= (min(h, w)//4)**2
        obj[mask] = contrast[1]
    return obj

def create_patterns(shape, N, scale=1, mask_contrast=1.0):
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    ref = ref * mask_contrast
    return ref

def add_noise_buckets(ref, obj, lambda_param=50, mu=0, sigma=1):
    N = ref.shape[2]
    buckets = np.zeros(N)
    for i in range(N):
        lam = lambda_param * np.sum(ref[:, :, i] * obj)
        shot = np.random.poisson(lam)
        thermal = np.random.normal(mu, sigma)
        buckets[i] = shot + thermal
    return buckets

def reconstruct(obj, ref, lambda_param=50, mu=0, sigma=1):
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)
    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]
    buckets = add_noise_buckets(ref_cropped, obj_cropped,
                                lambda_param=lambda_param, mu=mu, sigma=sigma)
    rec = np.mean(ref_cropped * buckets[np.newaxis, np.newaxis, :], axis=2) \
          - np.mean(ref_cropped, axis=2) * np.mean(buckets)
    return rec

def mse(a, b):
    return np.mean((a - b)**2)

# -----------------------------
# Experiment settings
# -----------------------------

object_shapes = ["square", "circle"]
object_sizes = [(50, 50), (100, 100)]
object_contrasts = [(0.0, 1.0), (0.2, 0.8)]
mask_contrasts = [0.5, 1.0, 2.0]
noise_params = [
    {"lambda_param": 10, "mu": 0, "sigma": 1},
    {"lambda_param": 50, "mu": 0, "sigma": 5},
    {"lambda_param": 100, "mu": 0, "sigma": 20}
]
N = 200
scale = 2

# -----------------------------
# Run experiments
# -----------------------------
results = []

for form in object_shapes:
    for size in object_sizes:
        for contrast in object_contrasts:
            obj = create_object(size, contrast=contrast, form=form)
            for mask_c in mask_contrasts:
                ref = create_patterns(size, N, scale=scale, mask_contrast=mask_c)
                for noise in noise_params:
                    rec = reconstruct(obj, ref,
                                      lambda_param=noise["lambda_param"],
                                      mu=noise["mu"],
                                      sigma=noise["sigma"])
                    error = mse(obj, rec)
                    results.append({
                        "shape": form,
                        "size": size,
                        "contrast": contrast,
                        "mask_contrast": mask_c,
                        "lambda_param": noise["lambda_param"],
                        "sigma": noise["sigma"],
                        "mse": error,
                        "reconstruction": rec
                    })

# -----------------------------
# Visualization: reconstructions
# -----------------------------
cols = 3
rows = int(np.ceil(len(results) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten()

for i, res in enumerate(results):
    ax = axes[i]
    ax.imshow(res["reconstruction"], cmap="gray")
    ax.axis("off")
    ax.set_title(f"{res['shape']}, size={res['size']}\n"
                 f"c={res['contrast']}, mC={res['mask_contrast']}\n"
                 f"λ={res['lambda_param']}, σ={res['sigma']}\nMSE={res['mse']:.2f}")

for ax in axes[len(results):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("reconstruction_all_variations.png", dpi=300)
plt.close()

# -----------------------------
# Convert results to DataFrame
# -----------------------------
df = pd.DataFrame([{
    "shape": r["shape"],
    "size": r["size"],
    "contrast": r["contrast"],
    "mask_contrast": r["mask_contrast"],
    "lambda": r["lambda_param"],
    "sigma": r["sigma"],
    "mse": r["mse"]
} for r in results])

# -----------------------------
# Visualization: MSE trends
# -----------------------------
plt.figure(figsize=(12,4))
sns.scatterplot(x="lambda", y="mse", hue="sigma", style="shape", size="mask_contrast", data=df, palette="viridis", s=100)
plt.title("MSE vs Lambda (Shot Noise), colored by Sigma (Thermal Noise), mask contrast size-coded")
plt.xlabel("Lambda (Shot noise)")
plt.ylabel("MSE")
plt.savefig("mse_vs_lambda.png", dpi=300)
plt.close()

plt.figure(figsize=(12,4))
sns.scatterplot(x=df["contrast"].apply(lambda x: x[1]), y="mse", hue="shape", size="mask_contrast", data=df, palette="coolwarm", s=100)
plt.title("MSE vs Object Contrast (max pixel value)")
plt.xlabel("Object Contrast (max)")
plt.ylabel("MSE")
plt.savefig("mse_vs_contrast.png", dpi=300)
plt.close()

plt.figure(figsize=(12,4))
sns.scatterplot(x="mask_contrast", y="mse", hue="shape", style="contrast", size="lambda", data=df, palette="magma", s=100)
plt.title("MSE vs Mask Contrast")
plt.xlabel("Mask Contrast")
plt.ylabel("MSE")
plt.savefig("mse_vs_mask_contrast.png", dpi=300)
plt.close()

"""
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# -----------------------------
# Functions
# -----------------------------

def create_object(shape, contrast=(0.0, 1.0), form="square"):
   """ """Create a grayscale object with specified contrast and shape.""" """
    obj = np.full(shape, contrast[0])
    h, w = shape
    if form == "square":
        obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    elif form == "circle":
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - h//2)**2 + (xx - w//2)**2 <= (min(h, w)//4)**2
        obj[mask] = contrast[1]
    return obj

def add_thermal_noise(img, sigma):
    """ """Add Gaussian noise to the object.""" """
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    noisy = np.clip(noisy, 0, 1)
    return noisy

# -----------------------------
# Parameters
# -----------------------------
obj_shape = (64, 64)
contrast = (0.0, 1.0)
form = "square"
noise_levels = [0.05, 0.2, 0.5, 0.8, 1.0]

# Create object
obj = create_object(obj_shape, contrast=contrast, form=form)

# -----------------------------
# Plot noisy images and histograms
# -----------------------------
fig, axes = plt.subplots(len(noise_levels), 2, figsize=(10, 12))

for i, sigma in enumerate(noise_levels):
    noisy_img = add_thermal_noise(obj, sigma)
    
    # Show noisy image
    axes[i, 0].imshow(noisy_img, cmap="gray")
    axes[i, 0].set_title(f"Noisy Image σ={sigma}")
    axes[i, 0].axis("off")
    
    # Show histogram of pixel values
    axes[i, 1].hist(noisy_img.ravel(), bins=50, color='gray')
    axes[i, 1].set_title(f"Pixel Distribution σ={sigma}")
    axes[i, 1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig("thermal_noise_examples.png", dpi=300)
plt.close()

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pandas as pd

# -----------------------------
# Functions
# -----------------------------

def create_object(shape, contrast=(0.0, 1.0), form="square"):
    """Create a grayscale object with specified contrast and shape."""
    obj = np.full(shape, contrast[0])
    h, w = shape
    if form == "square":
        obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    elif form == "circle":
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - h//2)**2 + (xx - w//2)**2 <= (min(h, w)//4)**2
        obj[mask] = contrast[1]
    return obj


def create_patterns(shape, N, scale=1, mask_contrast=1.0):
    """Create random patterns for GI with controlled mask contrast."""
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    ref = ref * mask_contrast
    return ref

def add_thermal_noise(img, sigma):
    """Add Gaussian (thermal) noise."""
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def add_shot_noise(img, lambda_param):
    """Add Poisson (shot) noise."""
    scaled_img = img * lambda_param
    noisy = np.random.poisson(scaled_img)
    noisy = noisy / lambda_param
    return np.clip(noisy, 0, 1)

def add_noise_buckets(ref, obj, lambda_param=50, mu=0, sigma=1):
    """Simulate GI bucket measurements with shot + thermal noise."""
    N = ref.shape[2]
    buckets = np.zeros(N)
    for i in range(N):
        lam = lambda_param * np.sum(ref[:, :, i] * obj)
        shot = np.random.poisson(lam)
        thermal = np.random.normal(mu, sigma)
        buckets[i] = shot + thermal
    return buckets

def reconstruct(obj, ref, lambda_param=50, mu=0, sigma=1):
    """Reconstruct object using co-variance formula."""
    h_obj, w_obj = obj.shape
    h_ref, w_ref, N = ref.shape
    h_min = min(h_obj, h_ref)
    w_min = min(w_obj, w_ref)
    obj_cropped = obj[:h_min, :w_min]
    ref_cropped = ref[:h_min, :w_min, :]
    buckets = add_noise_buckets(ref_cropped, obj_cropped,
                                lambda_param=lambda_param, mu=mu, sigma=sigma)
    rec = np.mean(ref_cropped * buckets[np.newaxis, np.newaxis, :], axis=2) \
          - np.mean(ref_cropped, axis=2) * np.mean(buckets)
    rec = np.clip(rec, 0, 1)
    return rec

def mse(a, b):
    return np.mean((a - b)**2)

# -----------------------------
# Experiment settings
# -----------------------------

object_shapes = ["square", "circle"]
object_sizes = [(64, 64), (128, 128)]
object_contrasts = [(0.0, 1.0), (0.2, 0.8)]
mask_contrasts = [0.5, 1.0, 2.0]

# Noise parameters
thermal_params = [0.05, 0.2, 0.5, 0.8]
shot_params = [5, 20, 50, 100]

N = 200
scale = 2

# -----------------------------
# Run Thermal Noise Visualization
# -----------------------------
obj = create_object((64, 64))
fig, axes = plt.subplots(len(thermal_params), 2, figsize=(10, 12))

for i, sigma in enumerate(thermal_params):
    noisy = add_thermal_noise(obj, sigma)
    axes[i, 0].imshow(noisy, cmap="gray")
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Thermal noise σ={sigma}")
    axes[i, 1].hist(noisy.ravel(), bins=50, color="gray")
    axes[i, 1].set_xlim(0, 1)
    axes[i, 1].set_title(f"Pixel histogram σ={sigma}")

plt.tight_layout()
plt.savefig("thermal_noise_examples.png", dpi=300)
plt.close()

# -----------------------------
# Run Shot Noise Visualization
# -----------------------------
fig, axes = plt.subplots(len(shot_params), 2, figsize=(10, 12))

for i, lam in enumerate(shot_params):
    noisy = add_shot_noise(obj, lam)
    axes[i, 0].imshow(noisy, cmap="gray")
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Shot noise λ={lam}")
    axes[i, 1].hist(noisy.ravel(), bins=50, color="gray")
    axes[i, 1].set_xlim(0, 1)
    axes[i, 1].set_title(f"Pixel histogram λ={lam}")

plt.tight_layout()
plt.savefig("shot_noise_examples.png", dpi=300)
plt.close()



