import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

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

def reconstruct(obj, ref):
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
    rec = np.clip(rec, 0, 1)
    return rec

# -----------------------------
# Experiment: Vary Mask Contrast
# -----------------------------
object_size = (128, 128)
object_contrast = (0.0, 1.0)
object_form = "square"
N = 2000
scale = 2
mask_contrasts = [0.2, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(1, len(mask_contrasts), figsize=(5*len(mask_contrasts), 5))
fig.suptitle("GI Reconstruction with Varying Mask Contrast", fontsize=16)

obj = create_object(object_size, contrast=object_contrast, form=object_form)

for i, mask_c in enumerate(mask_contrasts):
    ref = create_patterns(object_size, N, scale=scale, mask_contrast=mask_c)
    rec = reconstruct(obj, ref)
    
    ax = axes[i]
    im = ax.imshow(rec, cmap="viridis", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"Mask contrast {mask_c}", fontsize=12)

plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.savefig("reconstruction_varying_mask_contrast.png", dpi=300)
plt.close()
print("Saved reconstruction_varying_mask_contrast.png")
