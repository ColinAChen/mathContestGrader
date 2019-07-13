# mathContestGrader

Final project for my C and Assembly class during my senior year of high school. 

Designed to automaticall grade Carlmont High School Tuesday math contests by recognizing handwritten answers and comparing them to user inputted answers.

## What I learned 
* Programming in Python
* Using OpenCV for computer vision
* Using Keras with a Tensorflow backed to create a nerual network

## How does it work?

### Parsing a math contest
I took advantage of the fact that answers will always appear in the answer column on the right. I used OpenCV to apply a variety of computer vision techniques to extract the answers from the scanned image and process each extracted image to match the format of the training data.

### Creating a handwriting recognition algorithm
Recognizing handwritten digits is a trivial problem because of the MNIST dataset of handwritten digits. I followed a handful of online tutorials to implement a Convolutional Neural Network built with Keras on a Tensorflow backed to identify handwritten digits.

### Putting it all together
The script prompts the user for the correct answers. It then parses a contest specified within the script, isoaltes the numbers within each anser box, and attempts to center and resize them to match the training data's format. Finally, it makes a prediction of what digit the handwritten digit is and compares it to the correct answer. 

## For the future
It is unlikely I will work more on this project. However, there are serveral ways it could be improved.
* Use [argparser](https://docs.python.org/3/library/argparse.html) to allow the user to designate the image's path during runtime.
* Process non-scanned images by using feature matching to account for the angle of the camera.
* Recognize multi-digit answers by continuing to split images into individual digits or by training a new classifier.
* Recognize letters and mathematical symbols.


