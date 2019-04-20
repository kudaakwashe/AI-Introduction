
# coding: utf-8

# In[2]:


get_ipython().magic(u'matplotlib inline')
from matplotlib.pyplot import imshow
from PIL import Image
import requests
from io import BytesIO
import json

# set up API configuration
endpointUrl = "https://westus.api.cognitive.microsoft.com/luis/v2.0/apps/383a7f6f-25b4-49cd-b718-1e28ce629510?verbose=true&timezoneOffset=-360&subscription-key=e25d4c1eada84549b17fde24704e054f&q="

# prompt for a command
command = input('Please enter a command: \n')

# call LUIS service to get the JSON response
endpoint = endpointUrl + command.replace(" ","+")
response = requests.get(endpoint)
data = json.loads(response.content.decode("UTF-8"))

# Identify top scoring intent
intent = data["topScoringIntent"]["intent"]
if (intent == "Light On"):
    img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/LightOn.jpg'
elif (intent == "Light Off"):
    img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/LightOff.jpg'
else:
    img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/Dunno.jpg'

# get apt image and show it
response = requests.get(img_url)
img = BytesIO(response.content)
img = Image.open(img)
imshow(img)

