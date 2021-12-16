from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
import pickle

subscription_key = "ADD_KEY_HERE"
endpoint = "ADD_ENDPOINT_HERE"

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

count = 0
captions = pickle.load(open("ms_captions.pickle" , "rb"))
for id in captions:
    cap = captions[id]
    if (len(cap.captions) == 0):
        print("No description detected.")
        count += 1
    else:
        for caption in cap.captions:
            print(caption.text)
            break
            #print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))

for f in os.listdir("595/data/val2014"):
    image = open("595/data/val2014/" + f, "rb")
    description_result = computervision_client.describe_image_in_stream(image)
    print(description_result)

    captions[f] = description_result
    with open('ms_captions.pickle', 'wb') as handle:
        pickle.dump(captions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    count += 1

    # azure can only handle 20 calls/min, so once every 4 sec to be safe
    time.sleep(4)
    if count == 5000:
        break

with open('ms_captions.pickle', 'wb') as handle:
    pickle.dump(captions, handle, protocol=pickle.HIGHEST_PROTOCOL)