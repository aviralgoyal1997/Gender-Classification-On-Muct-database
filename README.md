# Gender-Classification-On-Muct-database
In this I will train my random forest with pca and gaussian discriminant with pca model on muct landmarks data.This database consist 
76 pixel coordinate values which are important facial features but if we gonna apply model on new data,how will be convert image 
so I am using dlib python for 68 landmarks detection in an image,then I am giving this as input to our model.As it is 68,so some work 
was done on training data to get same dimensions

Install muct database from :  http://www.milbo.org/muct/ for  more details of data

download http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 in ur current working directory,dib gonna use this for landmark detection.

For dlib  in ubuntu :   https://www.learnopencv.com/install-dlib-on-ubuntu/ 
