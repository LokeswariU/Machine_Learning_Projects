import cv2
import numpy as np
import dlib
import pandas as pd
import math
import cv2
import statistics
import os

for n in (0,10):
	path = "C:/Users/lokes/Documents/Second_sem/MachineLearning/Dataset_video/Fold5_part2/60/"+str(n)+".mov"
	video_name = n
	print("pp",path)
	cap = cv2.VideoCapture(path)
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("C:/Users/lokes/Documents/Second_sem/MachineLearning/Drowsiness_detection/shape_predictor_68_face_landmarks.dat")
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	print("width",width)
	print("height",height)

	rotateCode = None
	# if(width > height):
	# 	rotateCode = cv2.ROTATE_90_CLOCKWISE

	if(n == 0):

		std_EAR = 0
		std_MAR = 0
		std_MOE = 0
		std_pupil = 0
		mean_EAR = 0
		mean_MAR = 0
		mean_MOE = 0
		mean_pupil = 0

		normal = pd.DataFrame(columns = ['n_EAR','n_MAR','n_MOE','n_pupil'])
		average_image_count=10
		for j in range(0,100,average_image_count):
			cap.set(cv2.CAP_PROP_POS_MSEC,j)
			ret,frame= cap.read()
			
			gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			if(rotateCode != None):
				gray = cv2.rotate(gray, rotateCode)
			
			faces = detector(gray)
			
			for face in faces:
				x1 = face.left()
				y1 = face.top()
				x2 = face.right()
				y2 = face.bottom()

				landmarks = predictor(gray,face)

				thirty_seven = [landmarks.part(37).x,landmarks.part(37).y]
				thirty_eight = [landmarks.part(38).x,landmarks.part(38).y]
				thirty_nine = [landmarks.part(39).x,landmarks.part(39).y]
				forty = [landmarks.part(40).x,landmarks.part(40).y]
				forty_one = [landmarks.part(41).x,landmarks.part(41).y]
				forty_two = [landmarks.part(42).x,landmarks.part(42).y]
				eye_height = math.sqrt(pow((forty_two[0]-thirty_eight[0]),2)+pow((forty_two[1]-thirty_eight[1]),2))+math.sqrt(pow((forty_one[0]-thirty_nine[0]),2)+pow((forty_one[1]-thirty_nine[1]),2))
				eye_width = 2*math.sqrt(pow((forty[0]-thirty_seven[0]),2)+pow((forty[1]-thirty_seven[1]),2))
				EAR = eye_height/eye_width

				sixty_one = [landmarks.part(61).x,landmarks.part(61).y]
				sixty_three = [landmarks.part(63).x,landmarks.part(63).y]
				sixty_five = [landmarks.part(65).x,landmarks.part(65).y]
				sixty_seven = [landmarks.part(67).x,landmarks.part(67).y]
				MAR = math.sqrt(pow((sixty_five[0]-sixty_one[0]),2)+pow((sixty_five[1]-sixty_one[1]),2))/math.sqrt(pow((sixty_seven[0]-sixty_three[0]),2)+pow((sixty_seven[1]-sixty_three[1]),2))

				MOE = MAR/EAR

				pupil_area = pow((math.sqrt(pow((forty_one[0]-thirty_eight[0]),2)+pow((forty_one[1]-thirty_eight[1]),2))/2),2)*3.14
				p1_p2 = math.sqrt(pow((thirty_eight[0]-thirty_seven[0]),2)+pow((thirty_eight[1]-thirty_seven[1]),2))
				p2_p3 = math.sqrt(pow((thirty_nine[0]-thirty_eight[0]),2)+pow((thirty_nine[1]-thirty_eight[1]),2))
				p3_p4 = math.sqrt(pow((forty[0]-thirty_nine[0]),2)+pow((forty[1]-thirty_nine[1]),2))
				p4_p5 = math.sqrt(pow((forty_one[0]-forty[0]),2)+pow((forty_one[1]-forty[1]),2))
				p5_p6 = math.sqrt(pow((forty_two[0]-forty_one[0]),2)+pow((forty_two[1]-forty_one[1]),2))
				p6_p1 = math.sqrt(pow((thirty_seven[0]-forty_two[0]),2)+pow((thirty_seven[1]-forty_two[1]),2))

				pupil_perimeter = p1_p2+p2_p3+p3_p4+p4_p5+p5_p6+p6_p1

				pupil_circularity = (4*3.14*pupil_area)/pow(pupil_perimeter,2)
				normal = normal.append({'EAR': EAR,'MAR':MAR,'MOE':MOE,'pupil':pupil_circularity},ignore_index = True)

			resize = cv2.resize(gray, (640, 480))
			cv2.imshow("Frame",resize)
			key = cv2.waitKey(1)
			if(key == 27):
				break

		mean_EAR = sum(normal['EAR'])/average_image_count
		mean_MAR = sum(normal['MAR'])/average_image_count
		mean_MOE = sum(normal['MOE'])/average_image_count
		mean_pupil = sum(normal['pupil'])/average_image_count
		std_EAR = statistics.stdev(normal['EAR'])
		std_MAR = statistics.stdev(normal['MAR'])


		std_MOE = statistics.stdev(normal['MOE'])
		std_pupil = statistics.stdev(normal['pupil'])
	#print(mean_EAR,std_EAR)
	# if(n == 10):
	# 	rotateCode = None
	# 	if(width > height):
	# 		rotateCode = cv2.ROTATE_90_CLOCKWISE


	cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
	frameRate = cap.get(cv2.CAP_PROP_POS_MSEC)
	rate = int(frameRate)
	splt = int(rate/100)

	df = pd.DataFrame(columns = ['status','EAR','MAR','MOE','pupil','n_EAR','n_MAR','n_MOE','n_pupil'])

	for i in range(0,rate,splt):
		cap.set(cv2.CAP_PROP_POS_MSEC,i)
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		if(rotateCode != None):
			gray = cv2.rotate(gray, rotateCode)

		faces = detector(gray)
		for face in faces:
			x1 = face.left()
			y1 = face.top()
			x2 = face.right()
			y2 = face.bottom()

			landmarks = predictor(gray,face)

			thirty_seven = [landmarks.part(37).x,landmarks.part(37).y]
			thirty_eight = [landmarks.part(38).x,landmarks.part(38).y]
			thirty_nine = [landmarks.part(39).x,landmarks.part(39).y]
			forty = [landmarks.part(40).x,landmarks.part(40).y]
			forty_one = [landmarks.part(41).x,landmarks.part(41).y]
			forty_two = [landmarks.part(42).x,landmarks.part(42).y]

			eye_height = math.sqrt(pow((forty_two[0]-thirty_eight[0]),2)+pow((forty_two[1]-thirty_eight[1]),2))+math.sqrt(pow((forty_one[0]-thirty_nine[0]),2)+pow((forty_one[1]-thirty_nine[1]),2))
			eye_width = 2*math.sqrt(pow((forty[0]-thirty_seven[0]),2)+pow((forty[1]-thirty_seven[1]),2))
			EAR = eye_height/eye_width

			sixty_one = [landmarks.part(61).x,landmarks.part(61).y]
			sixty_three = [landmarks.part(63).x,landmarks.part(63).y]
			sixty_five = [landmarks.part(65).x,landmarks.part(65).y]
			sixty_seven = [landmarks.part(67).x,landmarks.part(67).y]

			MAR = math.sqrt(pow((sixty_five[0]-sixty_one[0]),2)+pow((sixty_five[1]-sixty_one[1]),2))/math.sqrt(pow((sixty_seven[0]-sixty_three[0]),2)+pow((sixty_seven[1]-sixty_three[1]),2))

			MOE = MAR/EAR

			pupil_area = pow((math.sqrt(pow((forty_one[0]-thirty_eight[0]),2)+pow((forty_one[1]-thirty_eight[1]),2))/2),2)*3.14
			p1_p2 = math.sqrt(pow((thirty_eight[0]-thirty_seven[0]),2)+pow((thirty_eight[1]-thirty_seven[1]),2))
			p2_p3 = math.sqrt(pow((thirty_nine[0]-thirty_eight[0]),2)+pow((thirty_nine[1]-thirty_eight[1]),2))
			p3_p4 = math.sqrt(pow((forty[0]-thirty_nine[0]),2)+pow((forty[1]-thirty_nine[1]),2))
			p4_p5 = math.sqrt(pow((forty_one[0]-forty[0]),2)+pow((forty_one[1]-forty[1]),2))
			p5_p6 = math.sqrt(pow((forty_two[0]-forty_one[0]),2)+pow((forty_two[1]-forty_one[1]),2))
			p6_p1 = math.sqrt(pow((thirty_seven[0]-forty_two[0]),2)+pow((thirty_seven[1]-forty_two[1]),2))

			pupil_perimeter = p1_p2+p2_p3+p3_p4+p4_p5+p5_p6+p6_p1

			pupil_circularity = (4*3.14*pupil_area)/pow(pupil_perimeter,2)
			if(n ==0 ):
				df = df.append({'status':0,'EAR': EAR,'MAR':MAR,'MOE':MOE,'pupil':pupil_circularity},ignore_index = True)
			if( n == 10 ):
				df = df.append({'status':1,'EAR': EAR,'MAR':MAR,'MOE':MOE,'pupil':pupil_circularity},ignore_index = True)
			#print("eye_width",eye_width)

	print(mean_EAR,std_EAR)

	normalized_EAR = [(x-mean_EAR)/std_EAR for x in df['EAR']]
	normalized_MAR = [(x-mean_MAR)/std_MAR for x in df['MAR']]
	normalized_MOE = [(x-mean_MOE)/std_MOE for x in df['MOE']]
	normalized_pupil = [(x-mean_pupil)/std_pupil for x in df['pupil']]

	df['n_EAR'] = pd.Series(normalized_EAR, index=df.index)
	df['n_MAR'] = pd.Series(normalized_MAR, index=df.index)
	df['n_MOE'] = pd.Series(normalized_MOE, index=df.index)
	df['n_pupil'] = pd.Series(normalized_pupil, index=df.index)
	print(len(df))
	df.to_csv('C:/Users/lokes/Documents/Second_sem/MachineLearning/fold.csv', mode='a', header=False,index=False)


