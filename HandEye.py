#! /usr/bin/env python

import cv2 as cv
import numpy as np
import pylab as pl
import milk
from sys import argv, exit
from math import sqrt
from subprocess import check_output
from getopt import getopt, GetoptError


def getContour(image):
    retval, image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    kern = cv.getStructuringElement(cv.MORPH_CROSS, (4, 4))
    try:
        contours, hierarchy = cv.findContours(cv.erode(image, kern), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if not np.array_equal(contour, max(contours, key=len)):
                cv.drawContours(image, [contour], 0, 0, -1)
                cv.drawContours(image, [contour], 0, 0, 2)
    except:
        print('[!] Contour detection error')
        exit(0)
    return max(contours, key=len)


def getSkin(image):
    image = cv.blur(image, (3,3))
    image = cv.cvtColor(image, cv.COLOR_BGR2YCR_CB)
    Ych = np.zeros(image.shape[0:2], dtype=np.uint8)
    CRch = np.zeros(image.shape[0:2], dtype=np.uint8)
    CBch = np.zeros(image.shape[0:2], dtype=np.uint8)
    Ych[:,:] = image[:,:,0]
    CRch[:,:] = image[:,:,1]
    CBch[:,:] = image[:,:,2]
    Y = np.zeros((256, 1), dtype=np.uint8)
    Y[25:245] = 255
    CR = np.zeros((256, 1), dtype=np.uint8)
    CR[140:180] = 255
    CB = np.zeros((256, 1), dtype=np.uint8)
    CB[77:135] = 255
    Ych = cv.LUT(Ych, Y)
    CRch = cv.LUT(CRch, CR)
    CBch = cv.LUT(CBch, CB)
    image = np.zeros(image.shape[0:2], dtype=np.uint8)
    image[:,:] = CRch[:,:] & CBch[:,:] & Ych[:,:]
    return image


def resizeImage(image, width=-1, height=-1, percent=-1):
    if percent != -1:
        height = int(image.shape[0] * percent)
        width = int(image.shape[1] * percent)
    elif height != -1:
        width = image.shape[1] * height / image.shape[0]
    elif width != -1:
        height = image.shape[0] * width / image.shape[1]
    image = cv.resize(image, (width, height))
    return image


def getDefects(contour):
    try:
        defection = sqrt(cv.arcLength(contour, False))
        approx_contour = cv.approxPolyDP(contour, defection, False)
        hull = cv.convexHull(approx_contour, returnPoints=False)
        convex_defects = cv.convexityDefects(approx_contour, hull)
    except:
        print('[!] Defects detection error')
        exit(0)
    return approx_contour, convex_defects


def drawImage(image, contour, defects, label):
    cv.putText(image, label, (10,70), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv.CV_AA)
    try:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(image, (x, y), (x + w, y + h), [127, 0, 0], 2)
        cv.drawContours(image, contour, -1, [127, 127, 127], 20)
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            cv.line(image, tuple(contour[s][0]), tuple(contour[e][0]), [0, 127, 127], 2)
            cv.line(image, tuple(contour[s][0]), tuple(contour[f][0]), [127, 127, 127], 2)
            cv.line(image, tuple(contour[f][0]), tuple(contour[e][0]), [127, 127, 127], 2)
            cv.circle(image, tuple(contour[s][0]), 5, [0, 255, 0], -1)
            cv.circle(image, tuple(contour[e][0]), 5, [0, 255, 0], -1)
            cv.circle(image, tuple(contour[f][0]), 5, [0, 0, 255], -1)
    except:
        print('[!] Drawing error')
        exit(0)
    return image


def getFVector(image):
    defects_area = 0
    total_d = 0
    binary_image = getSkin(image)
    contour = getContour(binary_image)
    approx_contour, convex_defects = getDefects(contour)
    num_defects = len(convex_defects)
    for i in range(convex_defects.shape[0]):
        s, e, f, d = convex_defects[i, 0]
        defects_area += cv.contourArea(np.array([approx_contour[s], approx_contour[e], approx_contour[f]]))
        total_d += d
    x, y, w, h = cv.boundingRect(approx_contour)
    d = total_d/max(w,h)
    defects_ratio = defects_area/cv.contourArea(contour)
    ratio = float(w)/float(h)
    moments = cv.moments(contour)
    humoments = cv.HuMoments(moments)
    mass_center = ((moments['m10']/moments['m00'] - x)/w, (moments['m01']/moments['m00'] - y)/h)
    vector = humoments[:,0].tolist() + [num_defects, defects_ratio, d, ratio, mass_center[0], mass_center[1]]
    return vector


def trainMe(directory):
    classes = []
    labels = []
    features = []
    print('[+] Reading files')
    gestures = check_output(['ls', directory]).split()
    print('[+] Extracting features')
    for gesture in gestures:
        classes.append(gesture)
        gesture_dir = directory + '/' + gesture
        files = check_output(['ls', gesture_dir]).split()
        for filename in files:
            file_dir = gesture_dir + '/' + filename
            labels.append(len(classes))
            image = cv.imread(file_dir)
            image = resizeImage(image, width=500)
            features.append(getFVector(image))
    features = np.array(features)
    labels = np.array(labels)

    print('[+] Training')
    classifier = milk.defaultclassifier()
    model = classifier.train(features, labels)

    print('[+] Cross validation')
    confusion_matrix, names = milk.nfoldcrossvalidation(features, labels, learner=classifier)
    print('[+] Accuracy %.2f' % (float(confusion_matrix.trace())/float(confusion_matrix.sum())))

    return model, classes


def testMe(directory, model, classes):
    print('[+] Testing files')
    files = check_output(['ls', directory]).split()
    for filename in files:
        file_dir = directory + '/' + filename
        image = cv.imread(file_dir)
        image = resizeImage(image, width=500)
        print('%s => %s' %(file_dir, classes[model.apply(getFVector(image))-1]))


try:
    if len(argv) < 3:
        raise Exception()
    opts, args = getopt(argv[1:], 't:i:')
except:
    print('Wrong parameters!\n-t\tTraining directory\n-i\tInput directory or "cam" for camera')
    exit(0)

for opt, arg in opts:
    if opt == '-t':
        train_dir = arg
    elif opt == '-i':
        input_dir = arg

model, classes = trainMe(train_dir)

if input_dir == 'cam':
    cv.namedWindow('HandEye')
    print('[+] Initializing cam input')
    capture = cv.VideoCapture(1)
    while True:
        retval, image = capture.read()
        image = resizeImage(image, width=500)
        try:
            binary_image = getSkin(image)
            contour = getContour(binary_image)
            if cv.contourArea(contour) < 10000:
                raise Exception()
            approx_contour, convex_defects = getDefects(contour)
            features = getFVector(image)
            image = drawImage(image, approx_contour, convex_defects, classes[model.apply(features)-1])
            cv.imshow('HandEye', image)
            key = cv.waitKey(10)
            if key == 1048603:
                break
        except:
            cv.putText(image, 'Nothing!', (10,70), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv.CV_AA)
            cv.imshow('HandEye', image)
            key = cv.waitKey(10)
            if key == 1048603:
                break
else:
    testMe(input_dir, model, classes)
