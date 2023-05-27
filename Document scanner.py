import cv2
import numpy as np
import time

cap=cv2.VideoCapture(0)
time.sleep(2)


def display_stacked_images(images):
    # Determine the maximum height and total width of the stacked images
    for i in range (0,len(images)):
        images[i] = cv2.resize(images[i],(images[i].shape[1]//2,images[i].shape[0]//2))

    max_height = max(image.shape[0] for image in images)
    total_width = sum(image.shape[1] for image in images)

    # Create a blank canvas to hold the stacked images
    stacked_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Calculate the starting x-coordinate for each image in the stack
    x_offset = 0
    for image in images:
        if image.ndim == 2:
            # Convert 1-channel image to 3 channels for stacking with RGB images
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # Convert 4-channel image to 3 channels for stacking
            image = image[:, :, :3]

        stacked_image[:, x_offset:x_offset+image.shape[1]] = image
        x_offset += image.shape[1]

    # Display the stacked image in a window
    cv2.imshow('Stacked Images', stacked_image)


def getcontours(img,imgContour):
    width, height = int(620/1.5),int(877/1.5)

    img_warp = img_contour.copy()

    contours, heirachy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    img_width= img.shape[1]
    img_height = img.shape[0]

    img_area = img_width*img_height
     
    
    for cont in contours:
        area = cv2.contourArea(cont)
        #print(img_area,area)
        if area>img_area//65:
            cv2.drawContours(imgContour,cont,-1,(0,0,255),7)
            para = cv2.arcLength(cont,True)
            points = cv2.approxPolyDP(cont,0.02*para,True)

    
            if len(points)==4:
                temp_points=[]
                for point in points:
                    temp_points.append(point[0].tolist())
                    cv2.circle(imgContour,tuple(point[0]),5,(0,255,0),-1)
                    
        
                #original_points= np.float32([[295,53],[527,13],[337,387],[580,345]])
                original_points = np.float32(temp_points)#.astype(np.float32)
                trasform_points = np.float32([[0,0],[width,0],[width,height],[0,height]])
                
                #print(points)
                matrix = cv2.getPerspectiveTransform(original_points,trasform_points)
                output = cv2.warpPerspective(img_warp,matrix,(width,height))
                cv2.imshow("rectified",output)
 
def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",150,255,empty)
cv2.createTrackbar("Threshold2","Parameters",150,255,empty)

while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    #frame = cv2.imread('3.jpg')
    img_contour = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame,(7,7),1)

    threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")
    canny_frame = cv2.Canny(blur_frame,threshold1,threshold2)

    kernal = np.ones((5,5))
    img_dil = cv2.dilate(canny_frame,kernal,iterations=1)

    getcontours(img_dil,img_contour)

    # ret, thresh = cv2.threshold(gray_frame,200,255,cv2.THRESH_BINARY)

    # contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    images=[frame,blur_frame,canny_frame,img_contour]
    cv2.imshow("con",img_contour)
    display_stacked_images(images)

    # cv2.imshow('img',img_contour)
 

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()

