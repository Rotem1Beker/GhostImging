import numpy as np
import matplotlib.pyplot as plt

# Create a simple grayscale object
def create_object(shape=(64,64), contrast=(0.0, 1.0)):
    obj = np.full(shape, contrast[0])
    h, w = shape
    obj[h//4:3*h//4, w//4:3*w//4] = contrast[1]
    return obj

# Add shot (Poisson) noise
def add_shot_noise(img, lambda_param):
    scaled = img * lambda_param
    noisy = np.random.poisson(scaled)
    noisy = noisy / lambda_param
    return np.clip(noisy, 0, 1)

# Object
obj = create_object()

# Different λ values
lambdas = [1, 5, 10, 15]

fig, axes = plt.subplots(len(lambdas), 2, figsize=(10, 12))
for i, lam in enumerate(lambdas):
    noisy_img = add_shot_noise(obj, lam)
    axes[i, 0].imshow(noisy_img, cmap="viridis")
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Shot Noise λ={lam}")
    
    axes[i, 1].hist(noisy_img.ravel(), bins=50, color="gray")
    axes[i, 1].set_xlim(0,1)
    axes[i, 1].set_title("Pixel histogram")

plt.tight_layout()
plt.savefig("shot_noise.png", dpi=300)
plt.show()
