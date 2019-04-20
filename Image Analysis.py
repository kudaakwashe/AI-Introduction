
# coding: utf-8

# In[1]:


visionURI = 'westus.api.cognitive.microsoft.com'
visionKey = 'd8d785245dd14bf7a6cedd33b01ba3a5'


# In[4]:


# Get image from a URL
get_ipython().magic(u'matplotlib inline')
from matplotlib.pyplot import imshow
from PIL import Image
import requests
from io import BytesIO

img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme2.jpg'

# Get the image and show it
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
imshow(img)


# In[10]:


def get_image_features(img_url):
    import http.client, urllib.request, urllib.parse, urllib.error, base64, json
    
    headers = {
        # Request headers.
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': visionKey,
    }
    
    params = urllib.parse.urlencode({
        # Request parameters. All of them are optional.
        'visualFeatures': 'Categories,Description,Color',
        'language': 'en',
    })
    
    body = "{'url':'" + img_url + "'}"
    
    try:
        # Execute the REST API call and get response.
        conn = http.client.HTTPSConnection(visionURI)
        conn.request("POST", "/vision/v1.0/analyze?%s" % params, body, headers)
        response = conn.getresponse()
        data = response.read()
        
        # 'data' contains the JSON response.
        parsed = json.loads(data)
        if response is not None:
            return parsed
        conn.close()
        
    except Exception as e:
        print('Error:')
        print(e)
        
jsonData = get_image_features(img_url)
desc = jsonData['description']['captions'][0]['text']
print(desc)


# In[13]:


# View full details returned
print(jsonData)#(json.dumps(jsonData, sort_keys=True, indent=2))


# In[14]:


img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/uke.jpg'

# Get the image and show it
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
imshow(img)
jsonData = get_image_features(img_url)
desc = jsonData['description']['captions'][0]['text']
print(desc)


# In[15]:


img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/soccer.jpg'

# Get the image and show it
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))
imshow(img)
jsonData = get_image_features(img_url)
desc = jsonData['description']['captions'][0]['text']
print(desc)

