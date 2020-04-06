import cv2
import numpy as np
# import matplotlib.pyplot as plt

# return the image in color optimized format with edge: white and others: black
def canny(image):
    grayscale= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #change to grayscale
    blur = cv2.GaussianBlur(grayscale, (5,5),0) #reduce noise in the grayscale image
    canny = cv2.Canny(blur, 50,150) #only display edges between 50 and 150 gradient to create edge;eg. 57 goes towards 50 so black and 140 goes toward 150 so white
    return canny

# the triangular region which in on focus is highlighted and the rest of the image is set to black

def region_of_interest(image):
    height = image.shape[0]
    print('shape:'+str(height))
    polygons = np.array([
        [(200, height), (1100, height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (255,255,255))
    # return white mask in triangle
    masked_image = cv2.bitwise_and(image,mask) #AND -> white mask and lane image
    return masked_image

#
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5)) # 704*(3/5) = 420
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# it averages and optimizes the blue line (lanes on focus) into a single smooth line
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4) # separate the co-ordinates of each line
        parameters = np.polyfit((x1, x2), (y1,y2), 1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])



# displays the lines into the original image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) #color RGB format; its inverse in cv2
    return line_image

# image = cv2.imread('test_image.jpg') #reading the image
# lane_image = np.copy(image) #creating a copy of the image
# canny_image = canny(lane_image)
# triangle_image = region_of_interest(canny_image) # create ROI of traingle vs white lines
# # Hough transform to detect the lane of the image
# lines = cv2.HoughLinesP(triangle_image,2,np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image,lines)
# line_image = display_lines(lane_image,averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# # show in cv2 plot library
# cv2.imshow("result", combo_image)
# cv2.waitKey(0) #display image for some time; 0 = keep displaying

# import the video
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read() # first value: blank(boolean) second: current frame
    canny_image = canny(frame)
    triangle_image = region_of_interest(canny_image) # create ROI of traingle vs white lines
    # Hough transform to detect the lane of the image
    lines = cv2.HoughLinesP(triangle_image,2,np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # show in cv2 plot library
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):#display image for some time; 0 = keep displaying..... 0xFF : when key input is q then break
        break
    elif cv2.waitKey(1) & (cv2.getWindowProperty("result", cv2.WND_PROP_VISIBLE) < 1): #close the window when window close button is clicked
        break
cap.release()  # release capture and close all the windows
cv2.destroyAllWindows()
# # show in matplotlib
# plt.imshow(combo_image)
# plt.show()

# cv2.imshow('result',canny_image)
# cv2.imshow("result", region_of_interest(triangle_image))
# cv2.imshow("result",triangle_image)

