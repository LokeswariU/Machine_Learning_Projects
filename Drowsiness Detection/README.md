Libraries needed to run the project:
====================================

dlib, opencv, matplotlib, pandas, numpy, statistics, math, sklearn, xgboost, collections

How to run the project:
=======================

1) If you want to run the project as it is with the data provided in the zip file, you can run each algorithm individually from their corresponding .py files. The only thing that needs to be changed is the data location in the main function of the code.

2) If you want to run your own data as test data then a lot of preprocessing needs to be done. First use the preproces the video you are trying to give by using the preprocessing_video.py file. This will give you a csv file with 8 features and a status class. To run the file preprocessing_video.py, you are going to need another file called shape_predictor_68_face_landmarks.dat. This is also present in the zip file. Just make sure to change the location of this file correspondingly in the preprocessing_video.py file before running it.

3) If you are using KNN, you can go ahead and use the data as it is just by changing the location in the main function of the code and by changing the test train split. This can be done by changing the index on which it is being split.

4) But if you are planning on using any of the tree based algorithms, you need to run wcss.py on the csv file that was generated to findout how many clusters are ideal for each feature to be discretized into. This can be identified from the elbow point graph.

5) After the no. of clusters for each feature have been decided, use them in the preprocess.py file to discretize the data and save it into a new csv file.

6) Now that the csv file has been generated, you can go ahead and change the location in the main function of the code and run the algorithms.
