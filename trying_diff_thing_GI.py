"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# פונקציה שמייצרת אובייקט מלאכותי
def create_object(shape):
    obj = np.zeros(shape)
    h, w = shape
    obj[h//4:3*h//4, w//4:3*w//4] = 1.0   # ריבוע באמצע
    return obj

# פונקציה שמייצרת דפוסי אור עם פיצ'רים בגודל scale
def create_patterns(shape, N, scale=1):
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    return ref

# שחזור לפי נוסחת הקו-וריאנס
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
    
    rec = np.mean(ref_cropped * test[np.newaxis, np.newaxis, :], axis=2) \
          - np.mean(ref_cropped, axis=2) * np.mean(test)
    return rec

# =====================
# ניסוי עם פרמטרים שונים
# =====================

object_sizes = [(50, 100), (100, 200), (200, 400)]
N_values = [100, 1000, 5000]
scales = [1, 4, 10]

total = len(object_sizes) * len(N_values) * len(scales)
fig, axes = plt.subplots(1, total, figsize=(3*total, 4))

if total == 1:
    axes = [axes]

idx = 0
for size in object_sizes:
    obj = create_object(size)
    for N in N_values:
        for scale in scales:
            ref = create_patterns(size, N, scale=scale)
            rec = reconstruct(obj, ref)
            ax = axes[idx]
            ax.imshow(rec, cmap='viridis')  # שינוי לצבעוני
            ax.axis('off')
            ax.set_title(f"Size={size}\nN={N}, scale={scale}")
            idx += 1

plt.tight_layout()
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# פונקציה שמייצרת אובייקט מלאכותי
def create_object(shape):
    obj = np.zeros(shape)
    h, w = shape
    obj[h//4:3*h//4, w//4:3*w//4] = 1.0   # ריבוע באמצע
    return obj

# פונקציה שמייצרת דפוסי אור עם פיצ'רים בגודל scale
def create_patterns(shape, N, scale=1):
    h, w = shape
    h_s, w_s = int(np.ceil(h/scale)), int(np.ceil(w/scale))
    ref_small = np.random.rand(h_s, w_s, N)
    ref = np.zeros((h, w, N))
    for i in range(N):
        ref[:, :, i] = zoom(ref_small[:, :, i], (h/h_s, w/w_s), order=0)
    return ref

# שחזור לפי נוסחת הקו-וריאנס
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
    
    rec = np.mean(ref_cropped * test[np.newaxis, np.newaxis, :], axis=2) \
          - np.mean(ref_cropped, axis=2) * np.mean(test)
    return rec

# =====================
# ניסוי עם פרמטרים שונים
# =====================

object_sizes = [(50, 100), (100, 200), (200, 400)]
N_values = [100, 1000, 5000]
scales = [1, 4, 10]

# מספר השורות = מספר הקומבינציות של (object_size, N)
rows = len(object_sizes) * len(N_values)
cols = len(scales)

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))

if rows == 1:
    axes = np.array([axes])  # שיהיה ניתן לאנדקס תמיד כ-[row, col]

row_idx = 0
for size in object_sizes:
    obj = create_object(size)
    for N in N_values:
        for col_idx, scale in enumerate(scales):
            ref = create_patterns(size, N, scale=scale)
            rec = reconstruct(obj, ref)
            ax = axes[row_idx, col_idx]
            ax.imshow(rec, cmap='viridis')
            ax.axis('off')
            ax.set_title(f"Size={size}, N={N}, scale={scale}")
        row_idx += 1

plt.tight_layout()
plt.show()
