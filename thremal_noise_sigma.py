import numpy as np
import matplotlib.pyplot as plt

# Create a simple grayscale object
def create_object(shape=(64,64), contrast=(0.0, 1.0)):
    obj = np.full(shape, contrast[0])
    h, w = shape
    obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    return obj

# Add thermal (Gaussian) noise
def add_thermal_noise(img, sigma,miu):
    noisy = img + np.random.normal(miu, sigma, img.shape)
    return np.clip(noisy, 0, 1)

# Object
obj = create_object()

# Different noise levels
sigmas = [0.05, 0.2, 0.5, 0.8]
miu = 0.5

fig, axes = plt.subplots(len(sigmas), 2, figsize=(10, 12))
for i, sigma in enumerate(sigmas):
    noisy_img = add_thermal_noise(obj, sigma,miu)
    axes[i, 0].imshow(noisy_img, cmap="viridis")
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Thermal Noise σ={sigma}")
    
    axes[i, 1].hist(noisy_img.ravel(), bins=50, color="gray")
    axes[i, 1].set_xlim(0,1)
    axes[i, 1].set_title("Pixel histogram")

plt.tight_layout()
plt.savefig("thermal_noise.png", dpi=300)
plt.show()
