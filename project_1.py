import numpy as np
import cv2 
import imutils
from os import listdir

def canPanorama(imageA,imageB):
    descriptor = cv2.xfeatures2d.SIFT_create()
    kpsA, featuresA = descriptor.detectAndCompute(imageA, None)
    kpsA_pt = []
    for i in kpsA:
        kpsA_pt.append(i.pt)
    kpsA_pt = np.float32(kpsA_pt)
    kpsB, featuresB = descriptor.detectAndCompute(imageB, None)
    kpsB_pt = []
    for i in kpsB:
        kpsB_pt.append(i.pt)
    kpsB_pt = np.float32(kpsB_pt)
    
    
    
    # MATCH THOSE KEYPOINTS AND DESCRIPTORS
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for i in rawMatches:
        if(len(i) == 2 and i[0].distance< i[1].distance * 0.75):
            matches.append((i[0].trainIdx,i[0].queryIdx))
    
    HG_ptsA = np.empty((len(matches),2))
    HG_ptsB = np.empty((len(matches),2))
    
    for i in range(len(matches)):
        #temp = matches[i]
        HG_ptsA[i,:] = np.float32(kpsA_pt[matches[i][1]])
        HG_ptsB[i,:] = np.float32(kpsB_pt[matches[i][0]])
    # CREATE HOMOGRAPHY MATRIX
    reprojThresh = 4.0
    (H, status) = cv2.findHomography(HG_ptsA, HG_ptsB, cv2.RANSAC,reprojThresh)
    mask = status.ravel().tolist()
    if(len(status)>20 and sum(mask)>0.5 * len(status) ):
        return True
    else:
        return False

def createPanorama(imageA,imageB):
    descriptor = cv2.xfeatures2d.SIFT_create()
    kpsA, featuresA = descriptor.detectAndCompute(imageA, None)
    kpsA_pt = []
    for i in kpsA:
        kpsA_pt.append(i.pt)
    kpsA_pt = np.float32(kpsA_pt)
    kpsB, featuresB = descriptor.detectAndCompute(imageB, None)
    kpsB_pt = []
    for i in kpsB:
        kpsB_pt.append(i.pt)
    kpsB_pt = np.float32(kpsB_pt)
    
    
    
    # MATCH THOSE KEYPOINTS AND DESCRIPTORS
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for i in rawMatches:
        if(len(i) == 2 and i[0].distance< i[1].distance * 0.75):
            matches.append((i[0].trainIdx,i[0].queryIdx))
    
    HG_ptsA = np.empty((len(matches),2))
    HG_ptsB = np.empty((len(matches),2))
    
    for i in range(len(matches)):
        #temp = matches[i]
        HG_ptsA[i,:] = np.float32(kpsA_pt[matches[i][1]])
        HG_ptsB[i,:] = np.float32(kpsB_pt[matches[i][0]])
    # CREATE HOMOGRAPHY MATRIX
    reprojThresh = 4.0
    (H, status) = cv2.findHomography(HG_ptsA, HG_ptsB, cv2.RANSAC,reprojThresh)
    #wrapping
    result = cv2.warpPerspective(imageA, H,
    			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    return result


images = list()
for file in listdir():
    if file.endswith(".png"):
        temp = cv2.imread(file,0)
        temp = imutils.resize(temp,width = 400)
        images.append(temp)
        
#create a matrix which indicates set of panorame images
            #condition1: number of matches > certain_threshold
            #condition2: number of inliers > certain_threshold
setPanorama = np.zeros((len(images),len(images)))

for i in range(len(images)):
    for j in range(len(images)):
        if (canPanorama(images[i],images[j]) == True ):
            setPanorama[i][j] = 1

d = 0
for i in range(len(images)):
    j = i+1
    while(j<len(images)):
        final = images[i]
        if(setPanorama[i][j] == 1):
            final = createPanorama(final,images[j])
            
        j = j+1
    filename = "file_%d.jpg"%d
    d = d+1
    cv2.imwrite(filename,final)
            
