import cv2,glob

all_images=glob.glob("./Practice-2/*.jpg")
# print(all_images)
detect=cv2.CascadeClassifier("haarcascade.xml")

for image in all_images:
    img=cv2.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detect.detectMultiScale(gray,1.1,5)
    
    for x,y,w,h in faces:
        final_img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
       
    cv2.imshow('Face Detection',final_img)  
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    