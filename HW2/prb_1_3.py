import glob
import cv2 as cv
import numpy as np

path = glob.glob("/Users/ibnefarabishihab/Desktop/Course materials /ME 592/HW2/eia/produced/*.png")
images = []
for img in path:
    n = cv.imread(img)
    n=cv.cvtColor(n, cv.COLOR_BGR2GRAY)
    images.append(n)

def zca_whitening_matrix(images):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    newImages=[]
    for X in images:
        sigma = np.cov(X) # [M x M]
        print("shape ",sigma.shape)
        # Singular Value Decomposition. X = U * np.diag(S) * V
        U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
        # Whitening constant: prevents division by zero
        epsilon = 1e-5
        # ZCA Whitening matrix: U * Lambda * U'
        ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
        cv.imshow("image",ZCAMatrix)
        newImages.append(ZCAMatrix)
        
    return newImages
zca_whitening_matrix(images)