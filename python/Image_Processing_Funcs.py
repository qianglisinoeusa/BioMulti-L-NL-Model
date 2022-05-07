# Functions from AlRegib
# More info link https://ghassanalregibdotcom.wordpress.com/course/

import numpy as np
import dippykit as dip
import matplotlib.pyplot as plt

def klt_blocks(X: np.ndarray, k: int, B: int) -> np.ndarray:
    # Create a new matrix where each row represents a position in a block
    # and each column represents a block
    X_padded = np.pad(X, ((0, X.shape[0] % B), (0, X.shape[1] % B)),
                      mode='symmetric')
    block_cts = (int(X_padded.shape[0] / B), int(X_padded.shape[1] / B))
    X_blocked = np.zeros((B * B, block_cts[0] * block_cts[1]))
    for i in range(block_cts[0]):
        for j in range(block_cts[1]):
            X_blocked[:, (block_cts[1] * i) + j] = \
                X_padded[B * i:B * (i + 1), B * j:B * (j + 1)].reshape(B * B)

    # Calculate a column vector of the means of each row in X_blocked
    row_means = np.mean(X_blocked, axis=1).reshape((X_blocked.shape[0], 1))
    # Broadcast subtract the row_means from X_blocked to get Y. This,
    # in effect, centers the distribution of values in each row at 0 for Y.
    Y = X_blocked - row_means
    # Create a covariance matrix (PHI) for image X_blocked
    PHI = (1 / Y.shape[1]) * Y @ Y.T
    # Calculate the eigenvectors of the covariance matrix (PHI). We want to
    # have the eigenvectors in the eigenvector matrix (U) sorted in
    # descending order of corresponding eigenvalue. There are two ways to do
    # this (Note: the sign on each eigenvector may vary between the two
    # methods):
    #
    # APPROACH 1:
    # We can use the eigh() function, which (for Hermitian matrices,
    # which PHI is) returns quantities in order of increasing eigenvalue. We
    # can simply flip the column order of the eigenvector matrix to get our
    # matrix in its desired form. The fact that PHI is symmetric and
    # positive semi-definite is important, as it ensures that all
    # eigenvalues will be non-negative.
    #
    # _, U = np.linalg.eigh(PHI)
    # U = U[:, ::-1]
    #
    # APPROACH 2:
    # We can use the svd() function, which computes the singular value
    # decomposition. For symmetric matrices, the first term returned by the
    # svd() function will contain columns which are the eigenvectors of the
    # input. All quantities are returned in order of decreasing singular
    # value. The singular values are the absolute values of the eigenvalues
    # of the input, but since our input (PHI) is symmetric positive
    # semi-definite, the singular values are just the eigenvalues.
    # For brevity, we'll use this approach:
    U, _, _ = np.linalg.svd(PHI, full_matrices=True, compute_uv=True)
    # Extract only the first k basis vectors of the U matrix. These
    # are the ones corresponding to the k largest eigenvalues. Set A to be a
    # matrix whose rows are these k basis vectors.
    A = U[:, 0:k].T
    # Apply the A matrix to Y (we must use Y because it's rows have mean 0)
    # to get the KLT of the columns of the image Y, denoted by Z.
    Y_KLT = A @ Y
    # Reverse the KLT transform (with some data loss from compression,
    # of course) by left matrix multiplying Y_KLT by the transpose of A to
    # obtain a reconstructed Y. In the case where k is as large as possible
    # (k=X.shape[0]) the transpose of A is also the inverse of A.
    Y_reconstructed = A.T @ Y_KLT
    # Reconstruct the image X by adding back the row means to Y_reconstructed.
    X_blocked_reconstructed = Y_reconstructed + row_means
    #
    X_reconstructed = np.zeros(X_padded.shape)
    for i in range(block_cts[0]):
        for j in range(block_cts[1]):
            X_reconstructed[B * i:B * (i + 1), B * j:B * (j + 1)] = \
                X_blocked_reconstructed[:, (block_cts[1] * i) + j] \
                    .reshape((B, B))
    X_reconstructed = X_reconstructed[0:X.shape[0], 0:X.shape[1]]
    return X_reconstructed

def DCT_basis(N):
    x = np.arange(0,N)
    x = np.expand_dims(x,0)
    C = np.cos((2*x.transpose()+1)@x*np.pi/(2*N))*np.sqrt(2/N)

    basis = np.zeros((N**2,N**2))
    c = 0
    for i in range(N):
        for j in range(N):
            temp1 = C[:,[i]]
            temp2 = C[:,[j]].transpose()
            basis[:,c] = (temp1@temp2).flatten()
            c+=1

    return basis

'''
demo:

im_1 = dip.im_read('images/airplane_downsample_gray_square.jpg')
im_2 = dip.im_read('images/brussels_downsample_gray_square.jpg')
im_float_1 = dip.im_to_float(im_1)
im_float_2 = dip.im_to_float(im_2)

B = 10
k_vec = np.arange(1,B**2)
E1 = np.zeros(len(k_vec))
E2 = np.zeros(len(k_vec))
# Iterate over each specified k value
for i in range(len(k_vec)):
    # Reconstruct the images using the klt_columns function defined above
    im_1_reconstructed = klt_blocks(im_float_1, k_vec[i], B)
    im_2_reconstructed = klt_blocks(im_float_2, k_vec[i], B)

    E1[i] = 1 - dip.MSE(im_float_1,im_1_reconstructed)/dip.MSE(im_float_1,np.zeros(im_1.shape))
    E2[i] = 1 - dip.MSE(im_float_2,im_2_reconstructed)/dip.MSE(im_float_2,np.zeros(im_2.shape))

    plt.subplot(3, 2, 1)
    plt.imshow(im_1, cmap='gray')
    plt.title('image 1')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(im_2, cmap='gray')
    plt.title('image 2')
    plt.axis('off')


    plt.subplot(3, 2, 3)
    plt.imshow(im_1_reconstructed, cmap='gray')
    plt.title('image 1 reconstruction')

    plt.axis('off')


    plt.subplot(3, 2, 4)
    plt.imshow(im_2_reconstructed, cmap='gray')
    plt.title('image 2 reconstruction')
    plt.axis('off')

    plt.subplot(3, 2, (5, 6))
    plt.xlim(k_vec[0],k_vec[-1])
    plt.ylim(np.min([E1[0],E2[0]])*100,100)
    plt.grid(b=True, which ="both")
    plt.plot(k_vec[:i], E1[:i]*100,'b')
    plt.plot(k_vec[:i], E2[:i]*100,'r')
    plt.title('Percentage of Reconstructed Energy')
    plt.ylabel('%Energy')
    plt.legend(['image 1','image 2'])
    plt.pause(0.1)

'''

def display_network(A):

    # Compute rows & cols
    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = int(np.ceil(np.sqrt(col)))
    m = int(np.ceil(col / n))
    image = np.zeros(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            tmp = A[:,k]
            cmin = np.min(tmp)
            cmax = np.max(tmp)
            if cmin==cmax:
                tmp = np.ones(tmp.shape)
            else:
                tmp = (tmp-cmin)/(cmax-cmin)

            image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                tmp.reshape(sz, sz)

            k += 1

    return image


def im2blocks(img, block_size=8):
    img_size = img.shape
    img_blocks = np.zeros((block_size, block_size, int(np.floor(img_size[0] * img_size[1] / block_size ** 2))))
    c = 0
    for i in range(0, img_size[0] - block_size, block_size):
        for j in range(0, img_size[1] - block_size, block_size):
            img_blocks[:,:,c] = img[i:i + block_size, j:j + block_size]
            c += 1
            img_blocks[:,:,c] = (img_blocks[:,:,c]-img_blocks[:,:,c].min())


    return img_blocks



def KL(img, block_size=8):
    img_blocks = im2blocks(img, block_size)
    img_vectors = np.reshape(img_blocks,(-1,img_blocks.shape[-1]))
    numSamples = img_blocks.shape[-1]
    img_mean = np.mean(img_vectors, axis=1, keepdims=True)
    img_vectors = img_vectors - img_mean
    Rxx = img_vectors @ img_vectors.T / numSamples
    V, D = np.linalg.eig(Rxx)
    ind = np.argsort(V)
    ind = ind[::-1]
    D = D[:, ind]

    ind = np.arange(0, block_size * block_size)
    ind = np.reshape(ind, (block_size, block_size))
    tmp = dip.zigzag_indices(ind.shape)
    ind = ind[tmp]
    ind = np.argsort(ind)
    D = D[:, ind]

    return D

'''
demo:

import dippykit as dip
import functions as f

block_size = 8
img1 = dip.im_read('images/brussels_downsample_gray_square.jpg')
img2 = dip.im_read('images/airplane_downsample_gray_square.jpg')

DCT_basis = f.DCT_basis(block_size)
dip.imshow(DCT_basis,cmap='gray')
D1 = f.KL(img1, block_size=block_size)
D2 = f.KL(img2, block_size=block_size)


dip.subplot(2,3,1)
dip.imshow(img1, cmap='gray')
dip.title('Image 1')

dip.subplot(2,3,2)
dip.imshow(f.display_network(D1), cmap='gray')
dip.title('KLT basis for image1')

dip.subplot(2,3,4)
dip.imshow(img2, cmap='gray')
dip.title('Image 2')

dip.subplot(2,3,5)
dip.imshow(f.display_network(D2), cmap='gray')
dip.title('KLT basis for image2')

dip.subplot(2,3,(3,6))
dip.imshow(f.display_network(DCT_basis), cmap='gray')
dip.title('DCT basis')

dip.show()
'''

