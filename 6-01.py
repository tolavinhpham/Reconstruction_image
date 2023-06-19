import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale, iradon

#Create an original image
image = np.zeros((512, 512))
rr, cc = np.ogrid[:512, :512]
circle = (rr - 256) ** 2 + (cc - 256) ** 2 < 100 ** 2
image[circle] = 1

# Perform the Radon transform
theta = np.linspace(0., 360., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

# Perform the reconstruction
reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
error = reconstruction_fbp - image

# Display the original image, sinogram, reconstructed image, and reconstruction difference
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
ax1.imshow(image, cmap=plt.cm.Greys_r)
ax1.set_title('Original')
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 360, 0, sinogram.shape[0]), aspect='auto')
ax2.set_title('Radon transform\n(Sinogram)')
ax3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax3.set_title('Reconstruction\nFiltered back projection with filter')
imkwargs = dict(vmin=-0.2, vmax=0.2)
ax4.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
ax4.set_title('Reconstruction difference\nFiltered back projection')
plt.show()