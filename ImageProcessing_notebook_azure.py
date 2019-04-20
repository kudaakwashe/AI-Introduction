
# coding: utf-8

# In[4]:


get_ipython().magic(u'matplotlib inline')
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import skimage.color as sc

get_ipython().system(u'curl -o img.jpg https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme2.jpg')

i = np.array(Image.open('img.jpg'))
imshow(i)


# In[5]:


type(i)


# In[6]:


i.dtype


# In[7]:


i.shape


# In[8]:


i_mono = sc.rgb2gray(i)
imshow(i_mono, cmap='gray')
i_mono.shape


# In[9]:


def im_hist(img):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.gca()
    ax.hist(img.flatten(), bins=256)
    plt.show()
    
im_hist(i_mono)


# In[10]:


def im_cdf(img):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.gca()
    ax.hist(img.flatten(), bins=256, cumulative=True)
    plt.show()
    
im_cdf(i_mono)


# In[11]:


#Equalisation
from skimage import exposure

i_eq = exposure.equalize_hist(i_mono)
imshow(i_eq, cmap='gray')


# In[12]:


im_hist(i_eq)
im_cdf(i_eq)


# In[13]:


#Denoising with filters
import skimage
i_n = skimage.util.random_noise(i_eq)
imshow(i_n, cmap='gray')


# In[14]:


#Use Gaussian Filter
def gauss_filter(im, sigma = 10):
    from scipy.ndimage.filters import gaussian_filter as gf
    import numpy as np
    return gf(im, sigma = sigma)
i_g = gauss_filter(i_n)
imshow(i_g, cmap='gray')


# In[15]:


def med_filter(im, size = 10):
    from scipy.ndimage.filters import median_filter as mf
    import numpy as np
    return mf(im, size = size)
i_m = med_filter(i_n)
imshow(i_m, cmap='gray')


# In[16]:


#Edge detection
def edge_sobel(image):
    from scipy import ndimage
    import skimage.color as sc
    import numpy as np
    image = sc.rgb2gray(image) #convert color image to gray scale
    dx = ndimage.sobel(image, 1) #horizontal derivative
    dy = ndimage.sobel(image, 0) #vertical derivative
    mag = np.hypot(dx, dy) #magnitude
    mag *= 255.0 / np.amax(mag) #normalize (Q&D)
    mag = mag.astype(np.uint8)
    return mag

i_edge = edge_sobel(i_m)
imshow(i_edge, cmap="gray")


# In[19]:


#Corner detection

def corner_harr(im, min_distance = 10):
    from skimage.feature import corner_harris, corner_peaks
    mag = corner_harris(im)
    return corner_peaks(mag, min_distance = min_distance)

harris = corner_harr(i_eq, 10)

def plot_harris(im, harris, markersize = 20, color = 'red'):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(figsize=(6, 6))
    fig.clf()
    ax = fig.gca()
    ax.imshow(np.array(im).astype(float), cmap="gray")
    ax.plot(harris[:, 1], harris[:, 0], 'r+', color = color, markersize=markersize)
    return 'Done'

plot_harris(i_eq, harris)

