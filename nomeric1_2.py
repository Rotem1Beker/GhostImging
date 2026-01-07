import numpy as np
import matplotlib.pyplot as plt

# ===== חלק א: פונקציות ל-LU תלת-אלכסונית =====
def LU_tridiagonal(A):
    """
    מפרק מטריצה תלת-אלכסונית A ל-L ו-U.
    L: מטריצה תחתונה עם אלכסון יחידה ואלכסון תחתון אחד
    U: מטריצה עליונה עם אלכסון עליון אחד ואלכסון ראשי
    """
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    
    # אלגוריתם פירוק LU תלת-אלכסוני
    U[0,0] = A[0,0]
    if n > 1:
        U[0,1] = A[0,1]
    
    for i in range(1, n):
        L[i,i-1] = A[i,i-1] / U[i-1,i-1]
        U[i,i] = A[i,i] - L[i,i-1]*U[i-1,i]
        if i < n-1:
            U[i,i+1] = A[i,i+1]
    
    return L, U

def det_tridiagonal(A):
    """ מחשב דטרמיננטה של מטריצה תלת-אלכסונית באמצעות LU """
    L, U = LU_tridiagonal(A)
    return np.prod(np.diag(U))

# ===== חלק ב: יצירת מטריצה תלת-אלכסונית 10x10 =====
N = 10
A = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)

# ===== חישוב גרף דטרמיננטה של (λI - A) =====
lambdas = np.linspace(-4, 0, 100)
det_vals = []

for lam in lambdas:
    det_vals.append(det_tridiagonal(lam*np.eye(N) - A))

# ===== הצגת גרף =====
plt.figure(figsize=(8,5))
plt.plot(lambdas, det_vals, label="det(λI - A)")
plt.xlabel("λ")
plt.ylabel("det(λI - A)")
plt.title("דטרמיננטה של (λI - A) באמצעות LU תלת-אלכסוני")
plt.grid(True)
plt.legend()
plt.show()
