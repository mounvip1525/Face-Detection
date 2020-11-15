import cv2

detect=cv2.CascadeClassifier('haarcascade.xml')
imp_img=cv2.VideoCapture('zukenberg.jpg')

#res will have a boolean value if it reads the image or not and img will have the pixel cordinates of the image
res,img=imp_img.read()

#converting to a grey scale img as the haarcascade classifier works only on grey images 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#to detect faces of different sizes input(grey scale img, resizing factor, min. neighbours) and returns x,y,width and height of the img
faces=detect.detectMultiScale(gray,1.3,5)

for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)


#For showing the image
cv2.imshow('Elon Musk',img)
cv2.waitKey(0) #Can close the window whenever u want, else if set to 600ms then it'll close on its own after certain time
imp_img.release() 
cv2.destroyAllWindows()

