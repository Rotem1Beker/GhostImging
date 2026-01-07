import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import least_squares
import pylops
from pylops import MatrixMult, FirstDerivative
from pylops.optimization.sparsity import splitbregman      

# 1. create the object
obj = np.identity(10); obj = np.kron(obj, np.fliplr(np.identity(5)))
# 2. create the refes
N = 50 # number of realizations 
ref = np.random.random([obj.shape[0],obj.shape[1],N])
# 3. create the tests
test = (ref*obj[:,:,np.newaxis]).sum(axis=(0,1))
# 4. reconstruct the object

fig, ax = plt.subplots(1,3,figsize = [10,5])
ax[0].imshow(obj, aspect='auto'); ax[0].set_title('Object')
ax[1].imshow(ref[:,:,0], aspect='auto'); ax[1].set_title('A single light pattern')
ax[2].plot(test[:100], '--o',); ax[2].set_title('test measurements (1 to 100)')
plt.show()


"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#ex_01
#1
height, width = 100, 200
obj = np.random.rand(height, width)


#2
N = 5000
ref = np.random.rand(height, width, N)

print(ref.shape) 
"""
