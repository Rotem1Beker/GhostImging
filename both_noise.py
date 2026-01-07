import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# יצירת אובייקט
# -----------------------------
def create_object(shape=(64,64), contrast=(0.0, 1.0)):
    obj = np.full(shape, contrast[0])
    h, w = shape
    obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    return obj

# -----------------------------
# רעש תרמי (Gaussian)
# -----------------------------
def add_thermal_noise(img, sigma, miu):
    noisy = img + np.random.normal(miu, sigma, img.shape)
    return np.clip(noisy, 0, 1)

# -----------------------------
# רעש פואסון (Shot)
# -----------------------------
def add_shot_noise(img, lambda_param):
    scaled = img * lambda_param
    noisy = np.random.poisson(scaled)
    noisy = noisy / lambda_param
    return np.clip(noisy, 0, 1)

# -----------------------------
# יצירת אובייקט
# -----------------------------
obj = create_object()

# -----------------------------
# ערכי פרמטרים
# -----------------------------
lambdas = [1, 5, 10]
mius = [0.0, 0.5, 1.0]
sigmas = [0.05, 0.2, 0.5]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# -----------------------------
# שורה 1: שינוי λ
# -----------------------------
for i, lam in enumerate(lambdas):
    noisy_img = add_shot_noise(obj, lam)
    noisy_img = add_thermal_noise(noisy_img, sigma=0.1, miu=0.5)
    axes[0, i].imshow(noisy_img, cmap="viridis")
    axes[0, i].axis("off")
    axes[0, i].set_title(f"λ={lam}")

# -----------------------------
# שורה 2: שינוי μ
# -----------------------------
for i, miu in enumerate(mius):
    noisy_img = add_shot_noise(obj, 20)
    noisy_img = add_thermal_noise(noisy_img, sigma=0.1, miu=miu)
    axes[1, i].imshow(noisy_img, cmap="viridis")
    axes[1, i].axis("off")
    axes[1, i].set_title(f"μ={miu}")

# -----------------------------
# שורה 3: שינוי σ
# -----------------------------
for i, sigma in enumerate(sigmas):
    noisy_img = add_shot_noise(obj, 20)
    noisy_img = add_thermal_noise(noisy_img, sigma=sigma, miu=0.5)
    axes[2, i].imshow(noisy_img, cmap="viridis")
    axes[2, i].axis("off")
    axes[2, i].set_title(f"σ={sigma}")

plt.tight_layout()
plt.savefig("noise_comparison.png", dpi=300)
plt.show()
