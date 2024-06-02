# Assignment 3: Introduction to Augmented Reality

## Objectives
	* `Identify how images are represented using 2D and 3D arrays.`
	* `Learn the representation of color channels in 3D arrays and the predominance of a certain color in an image.`
	* `Use Hough tools to search and find lines and circles in an image.`
	* `Use the results from the Hough algorithms to identify basic shapes.`
	* `Understand how objects can be selected based on their pixel locations and properties.`
	* `Address the presence of distortion / noise in an image.`
	* `Identify what challenges real-world images present over simulated scenes.`
  
## Assignment Overview
In this assignment, the methods implemented utilize Hough Transform and line finding.

### Problem 1: Traffic Light
Given a scene, find the state of each traffic light and traffic light position in a scene.
ps2.py method "traffic_light_detection" returns the traffic light center coordinates (x, y), i.e. (col, row)
and the color of the light that is activated (‘red’, ‘yellow’, or ‘green’).
<figure style="text-align: center">
    <b>ps2-1-a-1</b><br><img src="output/ps2-1-a-1.png" width="250" alt="ps2-1-a-1">
    <figcaption><b>Traffic Light (center) Coordinates and active light: (135, 120), 'green'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-1-a-2</b><br><img src="output/ps2-1-a-2.png" width="250" alt="ps2-1-a-2">
    <figcaption><b>Traffic Light (center) Coordinates and active light: (437, 249), 'green'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-1-a-3</b><br><img src="output/ps2-1-a-3.png" width="250" alt="ps2-1-a-3">
    <figcaption><b>Traffic Light (center) Coordinates and active light: (130, 381), 'yellow'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-1-a-4</b><br><img src="output/ps2-1-a-4.png" width="250" alt="ps2-1-a-4">
    <figcaption><b>Traffic Light (center) Coordinates and active light: (630, 480), 'red'</b></figcaption>
</figure>

---
### Problem 2: Traffic Signs per Scene
Given a scene, find the traffic sign and its position in a scene.
<figure style="text-align: center">
    <b>ps2-2-a-1</b><br><img src="output/ps2-2-a-1.png" width="250" alt="ps2-2-a-1">
    <figcaption><b>Coordinates and Traffic Sign: (245, 345), 'no entry'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-2-a-2</b><br><img src="output/ps2-2-a-2.png" width="250" alt="ps2-2-a-2">
    <figcaption><b>Coordinates and Traffic Sign: (549, 247), 'stop'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-2-a-3</b><br><img src="output/ps2-2-a-3.png" width="250" alt="ps2-2-a-3">
    <figcaption><b>Coordinates and Traffic Sign: (250, 400), 'construction'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-2-a-4</b><br><img src="output/ps2-2-a-4.png" width="250" alt="ps2-2-a-4">
    <figcaption><b>Coordinates and Traffic Sign: (750, 350), 'warning'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-2-a-5</b><br><img src="output/ps2-2-a-5.png" width="250" alt="ps2-2-a-5">
    <figcaption><b>Coordinates and Traffic Sign: (307, 182), 'yield'</b></figcaption>
</figure>

---
### Problem 2: Multi Sign Detection
Given a scene multiple traffic signs, find and identify each sign.
<figure style="text-align: center">
    <b>ps2-3-a-1</b><br><img src="output/ps2-3-a-1.png" width="250" alt="ps2-3-a-1">
    <figcaption><b>Traffic Signs: 'no entry', 'stop', 'construction'</b></figcaption>
</figure>
<figure style="text-align: center">
    <b>ps2-3-a-2</b><br><img src="output/ps2-3-a-2.png" width="250" alt="ps2-3-a-2">
    <figcaption><b>Traffic Sign: 'traffic', 'no entry', 'stop', 'yield', 'construction', 'warning'</b></figcaption>
</figure>

---
### Challenge Problems
Given 'real world' images, find and label signs.

<figure style="text-align: center">
    <b>ps2-5-a-1</b><br><img src="output/ps2-5-a-1.png" width="250" alt="ps2-5-a-1">
    <figcaption><b>Coordinates and Name: Warning: (242, 321)</b></figcaption>
</figure>

<figure style="text-align: center">
    <b>ps2-5-a-2</b><br><img src="output/ps2-5-a-2.png" width="250" alt="ps2-5-a-2">
    <figcaption><b>Coordinates and Name:No Entry: (286, 214)</b></figcaption>
</figure>

<figure style="text-align: center">
    <b>ps2-5-a-3</b><br><img src="output/ps2-5-a-3.png" width="250" alt="ps2-5-a-3">
    <figcaption><b>Coordinates and Name:No Entry: (193, 211)</b></figcaption>
</figure>

For the ps2-5-a-3 input image, the warning search function failed to capture the center 
of the warning sign, but the no entry search function was able to return a reasonable 
location for the center of that sign. Several elements of the warning search function 
affected the results for the third image. First, the threshold value for the “cv2.HoughLinesP” 
in the warning function was set high to best capture ps2-5-a-1 warning sign’s sides. 
Second, the warning function converted the image to HSV, then applied a filter for yellow. 
This yellow range was based on the yellow range of the warning sign in the first image. 
Third, the warning search function included a “cv2.medianBlur” kernel size of 13, because 
the ps2-5-a-1 included fern trees behind the warning sign. Without the blur, the “cv2.Canny” 
function was not able to discern the sign from trees. However, the blur made the edges of 
the sign of the ps2-5-a-3 image less sharp before passing the image to the “cv2.Canny” function.
In addition to “cv2.HoughLinesP”, which was used for warning signs, “cv2.HoughCircles” was 
used for the no entry signs. Also, both search functions applied a blur to deal with 
background noise, then a sharpening function was applied to crisp the edges of the image 
before applying “cv2.Canny”.


