# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
from pyzbar import pyzbar
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", required=True, help="path to the image file")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])


# convert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)
cv2.waitKey(0)

# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (1, 1))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
cv2.imshow("image", thresh)
cv2.waitKey(0)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)
cv2.imshow("image", closed)
cv2.waitKey(0)

# find the contours in the thresholded image
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# initialize a list to store detected barcodes
detected_barcodes = []

# loop over the detected contours
for c in cnts:
    # compute the rotated bounding box of the contour
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # extract the coordinates of the bounding box vertices
    x, y, w, h = cv2.boundingRect(c)

    # crop the region within the bounding box
    cropped_image = image[y:y + h, x:x + w]

    # decode barcodes in the cropped image
    barcodes = pyzbar.decode(cropped_image)

    # append the detected barcodes to the list
    detected_barcodes.extend(barcodes)

# check if any barcodes are detected
if len(detected_barcodes) > 0:
    # handle the detected barcodes
    for i, barcode in enumerate(detected_barcodes):
        barcode_data = barcode.data
        # barcode_type = barcode.type
        if i == 0:
            barcode_type = 'NumPatient'
        elif i == 1:
            barcode_type = 'NumDossier'

        # print barcode data and type
        print(f"Barcode {i + 1} Data:", barcode_data)
        print(f"Barcode {i + 1} Type:", barcode_type)

        # extract the cropped image from the original image
        cropped_image = image[y:y + h, x:x + w]

        # display the cropped image in a separate window
        # resize image
        cropped_image = cv2.resize(cropped_image, (100, 20), fx=2, fy=2)

        cv2.imshow(f"Cropped Barcode {i + 1}", cropped_image)

        cv2.waitKey(0)

    # wait for a key press and then close all open windows
    cv2.destroyAllWindows()
    # Convert the list of barcode data to a JSON string
    json_str = json.dumps([{"data": barcode.data.decode(
    ), "type": barcode.type} for barcode in detected_barcodes])

    # Write the JSON string to a file
    with open("barcodes_data.json", "w") as json_file:
        json_file.write(json_str)

    print("Barcode data saved to 'barcodes_data.json'.")
else:
    print("No barcodes detected.")
