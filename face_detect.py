import cv2
import numpy
import sys
import math
import os


def put_padding(image):	
	height, width = image.shape[:2]
	padding = int(max(width, height)/2)
	borderType = cv2.BORDER_CONSTANT
	value = [0,0,0]
	destination = cv2.copyMakeBorder(image, padding, padding, padding, padding, borderType, value=value)
	return destination, padding

def remove_padding(image, padding):
	height, width = image.shape[:2]
	originalHeight = height - (padding*2)
	originalWidth = width - (padding*2)

	borderType = cv2.BORDER_CONSTANT
	value = [0,0,0]
	
	new_image = image[padding:padding+originalHeight, padding:padding+originalWidth ]
	return new_image

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    result = cv2.warpAffine(image, rot_mat, (width, height))
    return result

def multiply_v(matrix, image):
	ret, matrix = cv2.threshold(matrix, 1, 255, cv2.THRESH_BINARY)
	matrix[matrix == 255] = 1
	matrix = cv2.multiply(image, matrix)
	return matrix

# Get user supplied values
# the name of the image inside the root (just and not the path)
imageName = sys.argv[1]

if not os.path.exists(imageName[:-4]):
	os.mkdir(imageName[:-4])

# Create the haar cascade
# the path of .xml of Face CascadeClassifier
cascPath = "haarcascade_frontalface_default.xml" 
faceCascade = cv2.CascadeClassifier(cascPath)

#face cascade params
params = [ 
#[neighbors, scale, minSize, keepQualityAfterRotation]
	[14, 1.5, 15, True],
	[14, 1.5, 15, False],
	[12, 1.5, 15, True],
	[12, 1.5, 15, False],
	[10, 1.5, 15, True],
	[10, 1.5, 15, False],
	[8, 1.5, 15, True],
	[8, 1.5, 15, False],
	[1, 1.5, 15, True],
	[1, 1.5, 15, False],
	
	[14, 1.3, 15, True],
	[14, 1.3, 15, False],
	[12, 1.3, 15, True],
	[12, 1.3, 15, False],
	[10, 1.3, 15, True],
	[10, 1.3, 15, False],
	[8, 1.3, 15, True],
	[8, 1.3, 15, False],
	[1, 1.3, 15, True],
	[1, 1.3, 15, False],

	[14, 1.1, 15, True],
	[14, 1.1, 15, False],
	[12, 1.1, 15, True],
	[12, 1.1, 15, False],
	[10, 1.1, 15, True],
	[10, 1.1, 15, False],
	[8, 1.1, 15, True],
	[8, 1.1, 15, False],
	[1, 1.1, 15, True],
	[1, 1.1, 15, False]
]
for (neighbors, scale, minSize, keepQualityAfterRotation) in params:
	maxSize = 100 #constant
	# Read the original image
	src = cv2.imread(imageName)
	#Put padding on image to rotate it freely
	image, paddingAdded = put_padding(src) #image which will be transformed

	if keepQualityAfterRotation:
		#original pixels must be saved to keep quality after iterations
		srcPadded, paddingAdded = put_padding(src)

	paramsString = "_n="+ str(neighbors) + "_s=" + str(scale) + "_mins=" + str(minSize) + "_maxs=" + str(maxSize) + "_kqar=" + str(keepQualityAfterRotation)
	print(paramsString)
	log = open(imageName[:-4] + "/" + imageName[:-4] + paramsString + ".txt", "w") #file where the log will be written

	totalFacesDetected = 0
	#all the angles that the faces can be inclined
	angles = range(40, -40, -5)
	for rotate in angles:
		#rotate the image to detect inclined faces
		image = rotate_image(image, rotate)
		#Convert image to gray scale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Detect faces in the image
		faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor = scale,
			minNeighbors = neighbors,
			minSize = (minSize, minSize),
			maxSize = (maxSize, maxSize)
		)
		totalFacesDetected += len(faces)

		#Write on the log file
		print '\trotating ' + str(rotate) + " Founded {0} faces!".format(len(faces))
		log.write('rotating ' + str(rotate) + " Founded {0} faces!".format(len(faces)) + "\n");

		#Draw filled rectangles each face detect in the iteration
		#That way these ones won't be considered in the next iteration
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), -1)
		
		#rotate image to the original 
		image = rotate_image(image, -rotate)
		
		if keepQualityAfterRotation:
			image = multiply_v(image, srcPadded)

	print('Total faces detected = ' + str(totalFacesDetected))
	log.write('Total faces detected = ' + str(totalFacesDetected));
	image = remove_padding(image, paddingAdded)
	cv2.imwrite(imageName[:-4] + "/" + imageName[:-4] + paramsString + "_total=" + str(totalFacesDetected) + ".jpg", image)
