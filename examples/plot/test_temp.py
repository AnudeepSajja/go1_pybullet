import pinocchio as pin 
import numpy as np

a = np.array([[5, -2],
              [2, 3],
              [-1, 4]])

Q, R = np.linalg.qr(a, mode= "complete")
print(Q.shape)
print(R.shape)