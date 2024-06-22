import pandas as pd
import os
import cv2
from datetime import datetime

global imgList , steeringList
countFolder = 0
count = 0
imgList = []
steeringList = []

myDirectory = os.path.join(os.getcwd(),'DataCollected')
print(myDirectory)

while os.path.exists(os.path.join(myDirectory,f'IMG{str(countFolder)}')):
    countFolder +=1
newPath = myDirectory + "/IMG"+str(countFolder)

os.makedirs(newPath)

def saveData(img,steering):
    global imgList , steeringList
    now = datetime.now()
    timestamp = str(datetime.timestamp(now)).replace('.','')
    print("timestamp = ",timestamp)
    fileName = os.path.join(newPath , f'Image_{timestamp}.jpg')
    cv2.imwrite(fileName,img)
    imgList.append(fileName)
    steeringList.append(steering)

def saveLog():
    global imgList , steeringList
    rawData = {'Image': imgList,
               'Steering': steeringList}
    df = pd.DataFrame(rawData)
    df.to_csv(os.path.join(myDirectory,f'log_{str(countFolder)}.csv'), index=False , header=False)
    print('Log saved')
    print('Total Images:',len(imgList))

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    for x in range(10):
        _,img = cap.read()
        saveData(img,0.5)
        cv2.waitKey(1)
        cv2.imshow("Image",img)
    saveLog()