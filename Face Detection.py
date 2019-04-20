
# coding: utf-8

# In[2]:


faceURI = "https://westus.api.cognitive.microsoft.com/face/v1.0"
faceKey = "7132873edb854e79be38b24902c85046"


# In[3]:


get_ipython().system(u'pip install cognitive_face')
get_ipython().system(u'pip install pillow')


# In[8]:


get_ipython().magic(u'matplotlib inline')
import requests
from io import BytesIO
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import cognitive_face as CF

# Set URI and Key
CF.BaseUrl.set(faceURI)
CF.Key.set(faceKey)

# Detect face in image
img_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme1.jpg'
result = CF.face.detect(img_url)

# Get ID of first face detected
face1 = result[0]['faceId']
print("Face 1:" + face1)

# Get image
response = requests.get(img_url)
img = Image.open(BytesIO(response.content))

# Add rectangles for each face found
color = "blue"

if result is not None:
    draw = ImageDraw.Draw(img)
    for currFace in result:
        faceRectangle = currFace["faceRectangle"]
        left = faceRectangle["left"]
        top = faceRectangle["top"]
        width = faceRectangle["width"]
        height = faceRectangle["height"]
        draw.line([(left,top), (left+width,top)], fill=color, width=5)
        draw.line([(left+width,top), (left+width,top+height)], fill=color, width=5)
        draw.line([(left+width,top+height), (left,top+height)], fill=color, width=5)
        draw.line([(left,top+height), (left,top)], fill=color, width=5)
        
# show image
imshow(img)


# In[11]:


# Get image to compare
img2_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme2.jpg'
response2 = requests.get(img2_url)
img2 = Image.open(BytesIO(response2.content))

#Detect faces in a comparison image
result2 = CF.face.detect(img2_url)

# Assume first face is the one we want to compare
face2 = result2[0]['faceId']
print("Face 2:" + face2)

def verify_face(face1, face2):
    # By default, assume the match is unverified
    verified = "Not Verified"
    color="red"
    
    if result2 is not None:
        # compare the comparison face to the original one we retrieved previously
        verify = CF.face.verify(face1, face2)
        
        #if there's amatch, set verified and change color to green
        if verify['isIdentical']==True:
            verified = "Verified"
            color="lightgreen"
        
        #Display the second face with a red rectangle if unverified, or green if verified
        draw = ImageDraw.Draw(img2)
        for currFace in result2:
            faceRectangle = currFace["faceRectangle"]
            left = faceRectangle["left"]
            top = faceRectangle["top"]
            width = faceRectangle["width"]
            height = faceRectangle["height"]
            draw.line([(left,top), (left+width,top)], fill=color, width=5)
            draw.line([(left+width,top), (left+width,top+height)], fill=color, width=5)
            draw.line([(left+width,top+height), (left,top+height)], fill=color, width=5)
            draw.line([(left,top+height), (left,top)], fill=color, width=5)

    # show image
    imshow(img2)
    
    #Display verification status and confidence level
    print(verified)
    print("Confidence Level: " + str(verify["confidence"]))

verify_face(face1, face2)


# In[13]:


# Get another image to compare
img2_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/graeme3.jpg'
response2 = requests.get(img2_url)
img2 = Image.open(BytesIO(response2.content))

#Detect faces in a comparison image
result2 = CF.face.detect(img2_url)

# Assume first face is the one we want to compare
face2 = result2[0]['faceId']
print("Face 2:" + face2)

verify_face(face1, face2)


# In[14]:


# Another
img2_url = 'https://raw.githubusercontent.com/MicrosoftLearning/AI-Introduction/master/files/satya.jpg'
response2 = requests.get(img2_url)
img2 = Image.open(BytesIO(response2.content))

#Detect faces in a comparison image
result2 = CF.face.detect(img2_url)

# Assume first face is the one we want to compare
face2 = result2[0]['faceId']
print("Face 2:" + face2)

verify_face(face1, face2)

