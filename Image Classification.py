
# coding: utf-8

# In[1]:


# Install the Custom Vision SDK
get_ipython().system(u' pip install azure-cognitiveservices-vision-customvision')


# In[13]:


PREDICTION_KEY = '0d653a83ec6d4fc89a856fb1f9d3a751'
ENDPOINT = 'https://westeurope.api.cognitive.microsoft.com/customvision/v3.0/Prediction/'
PROJECT_ID = '72b588d1-8ac0-4cbe-82b6-2bb8d58faf32'


# In[16]:


from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient


import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
get_ipython().magic(u'matplotlib inline')

# Use two test images
test_img1_url = 'http://www.pachd.com/free-images/food-images/apple-01.jpg'
test_img2_url = 'http://www.pachd.com/free-images/food-images/carrot-01.jpg'

# Create an instance of prediction service
predictor = CustomVisionPredictionClient(PREDICTION_KEY, endpoint=ENDPOINT)

# Get prediction for image 1
result1 = predictor.predict_image_url(PROJECT_ID, url=test_img1_url)
# The results include a prediction for each tag, descending order of probability - so we'll get
prediction1 = result1.predictions[0].tag_name + ": {0:.2f}%".format(result1.predictions[0].probability)

# Get prediction for image 2
result2 = predictor.predict_image_url(PROJECT_ID, url=test_img2_url)
# The results include a prediction for each tag, descending order of probability - so we'll get
prediction2 = result2.predictions[0].tag_name + ": {0:.2f}%".format(result2.predictions[0].probability)

# Download images and show them
response = requests.get(test_img1_url)
img1 = Image.open(BytesIO(response.content))

response = requests.get(test_img2_url)
img2 = Image.open(BytesIO(response.content))

# Create a figure
fig = plt.figure(figsize=(16, 8))

# Subplot for first image and its predicted class
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(img1)
a.set_title(prediction1)

# Subplot for second image and its predicted class
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(img2)
a.set_title(prediction1)

plt.show()

