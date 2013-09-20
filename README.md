HandEye
=======

HandEye is a modifiable hand gesture recognizer using supervised pattern recognition algorithms and image processing techniques.

****
**System Requirements:**
* This Python script can run on Linux only, so get a Linux or get out.
* Python (2.7.5)
* Milk (0.5.3)
* OpenCV (2.4.6)
* Matplotlib (1.3.0)
* Numpy (1.7.1)

****
**How to run this thing?**

It's a two phase process, training the classifier using some training sample pictures then asking it to classify some unknown class pictures.

2 parameters should be passed as following:

*  *-t training samples directory where every class enclosed in standalone sub directory.*

*  *-i input samples directory or cam for webcam input.*

****

**Examples:**

    ./HandEye.py -t ~/hand_gestures -i ~/test_pics
  
  That will tell the script to look for pictures inside ~/hand_gestures directory and use them to train the classifier, then it will look for any picture inside ~/test_pics directory and input them to be classified and print out the results to stdout.
  
    ./HandEye.py -t ~/hand_gestures -i cam
    
  Same as above but it will use the first addressed webcam feed as input. You should see a window showing the webcam feed overlaid with the result.
  
  
  Take a look at the *gestures* included to see the directory hierarchy needed to train the classifier.
  
****

**Contributors:**

* Husam Bilal (http://github.com/husam212)
* Zaid Malhis (http://github.com/Malhis)
