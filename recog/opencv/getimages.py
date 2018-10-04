import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model

act1 = input("Answer 1: ")
act2 = input("Answer 2: ")
act3 = input("Answer 3: ")
act4 = input("Answer 4: ")
act5 = input("Answer 5: ")
act6 = input("Answer 6: ")
actans = [act1,act2,act3,act4,act5,act6]
for r in range (0,6):
	actans[r] = int(actans[r])
#displays the given image with the given name until any key is pressed
def display (str, img):
	cv2.imshow('str', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#takes a scanned jpg or png math contest, can be any size

img = cv2.imread('camtest.png',cv2.IMREAD_GRAYSCALE) #contest
#display ('scanned contest', img)
coltemp = cv2.imread('answercol.png', cv2.IMREAD_GRAYSCALE) #answer column, hardcoded to be the correct size 
#for a 8.5x11 inch math contest
rows,cols = img.shape
#Based off a math contest 816 pixels wide by 1056 pixels tall (8.5x11 inches)
img = cv2.resize(img, (816,1056), interpolation = cv2.INTER_CUBIC)
display ('resized contest', img)
#find the answer column using template matching
w,h = coltemp.shape[::-1]
method = eval('cv2.TM_CCOEFF')
res = cv2.matchTemplate(img, coltemp, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
#draw rectangle around answer column
cv2.rectangle(img, top_left, bottom_right, 255, 2)
display ('location of answer column', img)
x,y = (top_left[0] + 5, bottom_right[1]-4) #starting poineeeeeeeeeeet for answer column
xadd = (bottom_right[0] - 3) - (top_left[0] + 3) #width of answer column
print ('starting x,y')
print (x)
print (y)

anscol = img[y:y+670, x:x+xadd] #take answer column from scanned contest
display ('initial answer column', anscol)
blur = cv2.GaussianBlur(anscol,(5,5),0) #smooth image
ret, anscolcop = cv2.threshold(blur, 240, 255, cv2.THRESH_BINARY_INV) #0 = black, 255 = white anscolcop = inverted
display ("working answer column", anscolcop)
#find contours of inverted binary answer column
anscol2, contours, hierarchy = cv2.findContours(anscolcop ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#set initial value of highest and lowest points
cnt = contours[0]
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
top = topmost[1]
bot = bottommost[1]
right = rightmost[0]
print ('starting top, bot, right')
print (top)
print (bot)
print (right)
bot = 100
'''if (rightmost[0] > 50):
			if (bottommost[1] > bot):
				bot = bottommost[1]'''
#for each set of contours, find the highest and lowest point to refine the column
for j in range(0,len(contours)):

		cnt = contours[j]
		topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
		bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
		rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
		if (topmost[1]<top):
			top = topmost[1]
		'''if (rightmost[0] > 50):
			right = rightmost[0]
			if (bottommost[1] > bot):
				bot = bottommost[1]'''
		if (rightmost[0] > 50):
			if (bottommost[1] > bot):
				bot = bottommost[1]

print ('final top, bot, right')
print (top)
print (bot)
print (right)
print ('height of each answer')

vert = bot-top
print ('height of col')
print (vert)
anscol = img[y+4:y+(vert), x:x+xadd]
print ('height of each box')
yadd = int(vert/6)
print (yadd)
#display ('he', anscol)

ans1 = anscol[4:yadd-4, 0:xadd-4]
ans2 = anscol[yadd+4:2*yadd-4, 0:xadd-4]
ans3 = anscol[2*yadd+4:3*yadd-4, 0:xadd-4]
ans4 = anscol[3*yadd+4:4*yadd-4, 0:xadd-4]
ans5 = anscol[4*yadd+4:5*yadd-8, 0:xadd-4]
ans6 = anscol[5*yadd+4:6*yadd-8, 0:xadd-4]
'''
ans1 = anscol[0:yadd, 0:xadd]
ans2 = anscol[yadd:2*yadd, 0:xadd]
ans3 = anscol[2*yadd:3*yadd, 0:xadd]
ans4 = anscol[3*yadd:4*yadd, 0:xadd]
ans5 = anscol[4*yadd:5*yadd, 0:xadd]
ans6 = anscol[5*yadd:6*yadd, 0:xadd]
'''
#crop image to the square region of the answer
frows,fcols = ans1.shape
frows1,fcols1 = ans5.shape
xsub = int((fcols-frows)/2)
xsub1 = int((fcols1-fcols1)/2)
print (frows)
print(fcols)
print (xsub)


ans1 = ans1[0:fcols,xsub:fcols+xsub]
ans2 = ans2[0:fcols,xsub:fcols+xsub]
ans3 = ans3[0:fcols,xsub:fcols+xsub]
ans4 = ans4[0:fcols,xsub:fcols+xsub]
ans5 = ans5[0:fcols1,xsub1:fcols1+xsub1]
ans6 = ans6[0:fcols1,xsub1:fcols1+xsub1]
display ('answer1', ans1)
display ('answer2', ans2)
display ('answer3', ans3)
display ('answer4', ans4)
display ('answer5', ans5)
display ('answer6', ans6)
'''
leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

#draw extreme points
cv2.circle(anscol, leftmost, 3, 255),
cv2.circle(anscol, rightmost, 3, 255)
cv2.circle(anscol, topmost, 3, 255)
cv2.circle(anscol, bottommost, 3, 255)
'''
#each box has a width of 350 and a height of 227 pixels from inkscape
#starting coordinates
'''



'''
#takes an answer image, centers it, and inverts the color to match the MNIST format


def format ( img ):
	#average the image, for each pixel it should set it to the average of pixels around it, 
	#reduce noise and should eliminate stray points and eraser marks
	#kernel = np.ones((5,5),np.float32)/25 
	#blur = cv2.filter2D(img,-1,kernel) #img is input image, dst is blurred output
	blur = cv2.GaussianBlur(img,(5,5),0) #blurred image to smooth it out
	#blur = cv2.GaussianBlur(blur,(5,5),0)
	#Push gray values to white if they are at least 235
	ret, ans1s = cv2.threshold(blur, 235, 255, cv2.THRESH_BINARY_INV) #0 = black, 255 = white #ans1s = inverted blurred iamge
	
	#find contours of inverted binary image
	im2, contours, hierarchy = cv2.findContours(ans1s ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#print (len(contours))
	#find most extreme points from all lists
	cnt = contours[0]
	leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
	rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
	topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
	bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
	left = leftmost[0]
	right = rightmost[0]
	top = topmost[1]
	bot = bottommost[1]
	print ('old lrtb')
	print (left)
	print (right)
	print (top)
	print (bot)
	for x in range(0,len(contours)):
		
		cnt = contours[x]
		
		if (left < leftmost[0]):
			left = leftmost[0]
		if (right > rightmost[0]):
			right = rightmost[0]
		if (top < topmost[1]):
			top = topmost[1]
		if (bot > bottommost[1]):
			bot = bottommost[1]
		
		'''if (i == 0):
			cnt = contours[0]
			leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
			rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
			topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
			bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
			left = leftmost[0]
			right = rightmost[0]
			top = topmost[1]
			bot = bottommost[1]
'''
		'''if (len(contours[x])>len(cnt)):
			cnt = contours[x]
 		'''
	print ('new lrtb')
	print (left)
	print (right)
	print (top)
	print (bot)

	#find extreme values
	''''leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
	rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
	topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
	bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])'''
	'''
	draw extreme points
	cv2.circle(ans1, leftmost, 3, (0,0,0))
	cv2.circle(ans1, rightmost, 3, (0,0,0))
	cv2.circle(ans1, topmost, 3, (0,0,0))
	cv2.circle(ans1, bottommost, 3, (0,0,0))
	'''
	#find the center
	print ('cenlr, centb, center')
	cenlr = int((right + left)/2)
	print (cenlr)
	centb = int((bot + top)/2)
	print (centb)
	center = (cenlr,centb);
	print (center)
	#cv2.circle(ans1, center, 3, (0,0,0))
	rows,cols = img.shape
	print (rows)
	print (cols)
	'''(rows/2)-cenlr
	(cols/2) - centb

	(cols/2) - centb'''
	print ('x and y shift amount')
	print ((cols/2) - centb)
	print ((rows/2)-cenlr)

	#centering by transformation
	Matrix = np.float32([[1,0,(cols/2) - cenlr],[0,1, (rows/2)-centb]])
	dst = cv2.warpAffine(ans1s,Matrix,(cols,rows))
	
	final = cv2.resize(dst,(28, 28), interpolation = cv2.INTER_LINEAR)
	display ('image', final)
	#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
	final = final.reshape(1,1,28,28).astype('float32')
	#display ('image', final)
	#print (final)
	return final;


'''
format(ans1)
format(ans2)
format(ans3)
format(ans4)
format(ans5)
format(ans6)
'''

#destination folder for converting from jpg to MNIST binary
pathTest = 'C:/Users/colin/recog/convert/test-images/0'
#testing data folder
pathTrain = 'C:/Users/colin/recog/convert/training-images/0'

'''try:
	#saving answers to traning folder
	cv2.imwrite(os.path.join(pathTest,'answer1.jpg'), format(ans1))
	cv2.imwrite(os.path.join(pathTest,'answer2.jpg'), format(ans2))
	cv2.imwrite(os.path.join(pathTest,'answer3.jpg'), format(ans3))
	cv2.imwrite(os.path.join(pathTest,'answer4.jpg'), format(ans4))
	cv2.imwrite(os.path.join(pathTest,'answer5.jpg'), format(ans5))
	cv2.imwrite(os.path.join(pathTest,'answer6.jpg'), format(ans6))
	#save images to testing folder
	cv2.imwrite(os.path.join(pathTrain,'answer1.jpg'), format(ans1))
	cv2.imwrite(os.path.join(pathTrain,'answer2.jpg'), format(ans2))
	cv2.imwrite(os.path.join(pathTrain,'answer3.jpg'), format(ans3))
	cv2.imwrite(os.path.join(pathTrain,'answer4.jpg'), format(ans4))
	cv2.imwrite(os.path.join(pathTrain,'answer5.jpg'), format(ans5))
	cv2.imwrite(os.path.join(pathTrain,'answer6.jpg'), format(ans6))
	print ('images saved')
except: 
	print ("An error occured, not all images were saved")'''

#testing answer images
'''
img[0:218, 0:320] = (ans1) #hight, width
img[218:436, 0:320] = (ans2)
img[436:654, 0:320] = (ans3)
img[652:868, 0:320] = (ans4)
img[868:1086, 0:320] = (ans5)
img[1086:1304, 0:320] = (ans6)

'''
#a = format(ans1)
#a = a.reshape((1,1,28,28))
#print (type(a))
model = load_model('mymodel.h5')
#predict the value of a given answer x
def score(x):

	answer = model.predict(x)

	print (len(answer))
	print (answer)
	print (answer[0])
	print (answer[0][0])
	most = answer[0][0]
	count = 0.0
	for p in range (0,10):
		
		#print (count)
		#print (answer[0][p])
		if (answer[0][p]>most):
			#print ('logic works')
			most = answer[0][p]
			finalanswer = count
		count = count+1
	#finalanswer = answer[0].index(most)
	print ('final answer')
	print (finalanswer)
	return finalanswer
fn1 = score (format(ans1))
fn2 = score (format(ans2))
fn3 = score (format(ans3))
fn4 = score (format(ans4))
fn5 = score (format(ans5))
fn6 = score (format(ans6))
fnans = [fn1, fn2, fn3, fn4, fn5, fn6]
def grade (actual, test):
	finalscore = 0
	print ("Actual answers: ", actual)
	print ("Test answers: ", test)
	for g in range(0,6):
		if (actual[g] == test[g]):
			finalscore = finalscore + 1
	
	print ('final score: ', finalscore)
	return finalscore
grade (actans, fnans)