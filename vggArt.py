######/  FACES packages   /######
import numpy as np
from numpy import linalg as LA
import sys
import math
import csv
import gc
import os
from PIL import Image
from scipy.stats import norm
import cv2
import itertools
import scipy
from imutils import face_utils
import imutils
import dlib
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
from imutils import face_utils
import imutils
import dlib
os.environ['GLOG_minloglevel'] = '2'
import caffe
#################################
 

caffe.set_mode_cpu()

folder_index = 0

model_def = "./VGG_FACE_deploy.prototxt"
model_weights = "./styled_iter_45000.caffemodel" #VGG_ART_iter_45000.caffemodel



def faceCrop(img,shape_predictor_path = "./shape_predictor_68_face_landmarks.dat", crop_flag = True):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load the input image, resize it, and convert it to grayscale
    #image = cv2.imread(img_path)
    image = img
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    #cv2.imshow("Input", image)
    rects = detector(gray, 2)
    img_out = image
    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        try:
            (x, y, w, h) = rect_to_bb(rect)
            faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
            try:
                faceAligned = fa.align(image, gray, rect)   # initialize dlib's face detector (HOG-based) and then create
                img_out = faceAligned
            except:
                img_out = faceOrig
        except:
            img_out = image
    
    return img_out

def vgg_art(path_test_image,path_to_reference_image):
    #Distribution data from training set
    mean = 138.919
    std = 96.1188
    mean1 = 235.018
    std1 =  34.9898
    delta_u = 0.55
    delta_l = 0.45

    net = caffe.Net(model_def,      # defines the structure of the model
        model_weights,  # contains the trained weights
        caffe.TEST)     # use test mode (e.g., don't perform dropout)

    net_p = caffe.Net(model_def,      # defines the structure of the model
        model_weights,  # contains the trained weights
        caffe.TEST)     # use test mode (e.g., don't perform dropout)

    im = cv2.imread(path_test_image)
    im = faceCrop(im)
    im = cv2.resize(im, (224, 224))
    im = np.swapaxes(im,0,2)
    im = np.swapaxes(im,1,2)

    im_input = im[np.newaxis, :, :]
    net.blobs['data'].reshape(*im_input.shape)
    net.blobs['data'].data[...] = im_input
    output = net.forward()
    val_fc7 = net.blobs['fc7'].data

    im_p = cv2.imread(path_to_reference_image)
    im_p = faceCrop(im_p)
    im_p = cv2.resize(im_p, (224, 224))
    im_p = np.swapaxes(im_p,0,2)
    im_p = np.swapaxes(im_p,1,2)

    im_p_input = im_p[np.newaxis, :, :]
    net_p.blobs['data'].reshape(*im_p_input.shape)
    net_p.blobs['data'].data[...] = im_p_input
    output = net_p.forward()
    val_fc7_p = net_p.blobs['fc7'].data    

    delta_val_fc7 = val_fc7-val_fc7_p

    delta_chiSq = [ 0 if f1 == 0 and f2 == 0 else (f1-f2)**2/(f1+f2) for f1,f2 in zip(val_fc7[0],val_fc7_p[0])]

    chi_sim = sum(delta_chiSq)**0.5

    val_sim = LA.norm(delta_val_fc7, 2)
    prob_sim = scipy.stats.norm(mean, std).pdf(val_sim)
    prob_dis = scipy.stats.norm(mean1, std1).pdf(val_sim)

    hp_test = prob_sim/(prob_sim + prob_dis)

    if hp_test > delta_u :
        t_decision = "Match"
    elif hp_test< delta_l:
        t_decision = "Non-Match"
    else:
        t_decision = "Equivocal"

    return 'Decision: {0} with similarity: {1}'.format(t_decision,100*hp_test)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Test adn Reference Images')
    parser.add_argument("-t", "--test_img",\
                    help="path to test image file")
    parser.add_argument("-r", "--ref_img",\
                    help="path to reference image file")

    args = parser.parse_args()

    if len(sys.argv) != 2*len(vars(args)) +1 :
        sys.exit('[Error] Usage: python {0} -t <path of test image> -r <path of reference image>'.format(sys.argv[0]))

    path_test_img = args.test_img
    path_ref_image = args.ref_img
    print(vgg_art(path_test_img,path_ref_image))