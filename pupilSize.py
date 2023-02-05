import argparse
import os
import time

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

import mediapipe as mp

from config import config
from logger import Logger
from models import Simple, NASNET, Inception, GAP, YOLO
from utils import annotator, change_channel, gray_normalizer

import streamlit as st

tf.disable_v2_behavior()
mp_face_mesh = mp.solutions.face_mesh

def load_model(session, m_type, m_name, logger):
    # load the weights based on best loss
    best_dir = "best_loss"

    # check model dir
    model_path = "models/" + m_name
    path = os.path.join(model_path, best_dir)
    if not os.path.exists(path):
        raise FileNotFoundError

    if m_type == "simple":
        model = Simple(m_name, config, logger)
    elif m_type == "YOLO":
        model = YOLO(m_name, config, logger)
    elif m_type == "GAP":
        model = GAP(m_name, config, logger)
    elif m_type == "NAS":
        model = NASNET(m_name, config, logger)
    elif m_type == "INC":
        model = Inception(m_name, config, logger)
    else:
        raise ValueError

    # load the best saved weights
    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.log('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)

    else:
        raise ValueError('There is no best model with given model')

    return model


def rescale(image):
    """
    If the input video is other than network size, it will resize the input video
    :param image: a frame form input video
    :return: scaled down frame
    """
    scale_side = max(image.shape)
    # image width and height are equal to 192
    scale_value = config["input_width"] / scale_side

    # scale down or up the input image
    scaled_image = cv2.resize(image, dsize=None, fx=scale_value, fy=scale_value)

    # convert to numpy array
    scaled_image = np.asarray(scaled_image, dtype=np.uint8)

    # one of pad should be zero
    w_pad = int((config["input_width"] - scaled_image.shape[1]) / 2)
    h_pad = int((config["input_width"] - scaled_image.shape[0]) / 2)

    # create a new image with size of: (config["image_width"], config["image_height"])
    new_image = np.ones((config["input_width"], config["input_height"]), dtype=np.uint8) * 250

    # put the scaled image in the middle of new image
    new_image[h_pad:h_pad + scaled_image.shape[0], w_pad:w_pad + scaled_image.shape[1]] = scaled_image

    return new_image


def upscale_preds(_preds, _shapes):
    """
    Get the predictions and upscale them to original size of video
    :param preds:
    :param shapes:
    :return: upscales x and y
    """
    # we need to calculate the pads to remove them from predicted labels
    pad_side = np.max(_shapes)
    # image width and height are equal to 384
    downscale_value = config["input_width"] / pad_side

    scaled_height = _shapes[0] * downscale_value
    scaled_width = _shapes[1] * downscale_value

    # one of pad should be zero
    w_pad = (config["input_width"] - scaled_width) / 2
    h_pad = (config["input_width"] - scaled_height) / 2

    # remove the pas from predicted label
    x = _preds[0] - w_pad
    y = _preds[1] - h_pad
    w = _preds[2]

    # calculate the upscale value
    upscale_value = pad_side / config["input_height"]

    # upscale preds
    x = x * upscale_value
    y = y * upscale_value
    w = w * upscale_value

    return x, y, w


# load a the model with the best saved state from file and predict the pupil location
# on the input video. finaly save the video with the predicted pupil on disk
def main2(m_type, m_name, logger, video_path=None, write_output=True):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)

        # check input source is a file or camera
        if video_path == None:
            video_path = 0

        # load the video or camera
        cap = cv2.VideoCapture(video_path)
        ret = True
        counter = 0
        tic = time.time()
        frames = []
        preds = []

        while ret:
            ret, frame = cap.read()

            if ret:
                # Our operations on the frame come here
                frames.append(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                f_shape = frame.shape
                if frame.shape[0] != 192:
                    frame = rescale(frame)

                image = gray_normalizer(frame)
                image = change_channel(image, config["input_channel"])
                [p] = model.predict(sess, [image])
                x, y, w = upscale_preds(p, f_shape)

                preds.append([x, y, w])
                # frames.append(gray)
                counter += 1

        toc = time.time()
        print("{0:0.2f} FPS".format(counter / (toc - tic)))

    # get the video size
    video_size = frames[0].shape[0:2]
    if write_output:
        # prepare a video write to show the result
        video = cv2.VideoWriter("predicted_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                (video_size[1], video_size[0]))

        for i, img in enumerate(frames):
            labeled_img = annotator((0, 250, 0), img, *preds[i])
            video.write(labeled_img)

        # close the video
        cv2.destroyAllWindows()
        video.release()
    print("Done...")

def main(m_type, m_name, logger, video_path=None, write_output=True):
    with tf.Session() as sess:

        # load best model
        model = load_model(sess, m_type, m_name, logger)

        # check input source is a file or camera
        if video_path == None:
            video_path = 0

        # load the video or camera
        #cap = cv2.VideoCapture(video_path)
        ret = True
        counter = 0
        #tic = time.time()
        frames = []
        preds = []

        #while ret:
        #    ret, frame = cap.read()

        #    if ret:
                # Our operations on the frame come here
        #        frames.append(frame)
        frame = cv2.imread("image1.jpg")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_shape = frame.shape
        print("shape=", frame.shape)
        if frame.shape[0] != 192:
                    frame = rescale(frame)

        print("shape2=", frame.shape)
        cv2.imwrite("processed1.jpg", frame)
        image = gray_normalizer(frame)
        image = change_channel(image, config["input_channel"])
        [p] = model.predict(sess, [image])
        print("predictions1=", p)
        x, y, w = upscale_preds(p, f_shape)

        print("predictions2=", x,y,w)
        cv2.ellipse(frame, ((p[0], p[1]), (p[2], p[2]), 0), (0, 250, 250), 1)

        cv2.imwrite("processed4.jpg", frame)
        preds.append([x, y, w])
                # frames.append(gray)
        counter += 1

        #toc = time.time()
        #print("{0:0.2f} FPS".format(counter / (toc - tic)))

    '''# get the video size
    video_size = frames[0].shape[0:2]
    if write_output:
        # prepare a video write to show the result
        video = cv2.VideoWriter("predicted_video.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                (video_size[1], video_size[0]))

        for i, img in enumerate(frames):
            labeled_img = annotator((0, 250, 0), img, *preds[i])
            video.write(labeled_img)

        # close the video
        cv2.destroyAllWindows()
        video.release()
    '''
    print("Done...")

def run_me(cv2_img, model, sess):
    # read image from camera


            preds = []

            # while ret:
            #    ret, frame = cap.read()

            #    if ret:
            # Our operations on the frame come here
            #        frames.append(frame)
            frame = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            f_shape = frame.shape
            print("shape=", frame.shape)
            if frame.shape[0] != 192:
                frame = rescale(frame)

            #print("shape2=", frame.shape)
            #cv2.imwrite("processed1.jpg", frame)
            image = gray_normalizer(frame)
            image = change_channel(image, config["input_channel"])
            [p] = model.predict(sess, [image])
            #print("predictions1=", p)
            x, y, w = upscale_preds(p, f_shape)

            #print("predictions2=", x, y, w)
            cv2.ellipse(frame, ((p[0], p[1]), (p[2], p[2]), 0), (0, 250, 250), 1)

            #cv2.imwrite("processed4.jpg", frame)
            preds.append([x, y, w])
            # frames.append(gray)
            #st.write("predictions=",x,y,w)
            return frame
            #st.image(frame ,width=500)



def cut_image(image, start_point, end_point):
        start = (int(landmarks[start_point][0] * cv2_img.shape[1]),
                 int(landmarks[start_point][1] * cv2_img.shape[0]))
        #st.write("start=",start)
        #cv2.circle(image, start, 5, (255,0,0), -1)

        end = (int(landmarks[end_point][0] * cv2_img.shape[1]),
               int(landmarks[end_point][1] * cv2_img.shape[0]))
        #cv2.circle(image, end, 5, (255, 0, 0), -1)
        #st.write("end=", end)

        cv2.rectangle(image, start, end, (255,0,0), 3)
        eye = cv2_img[start[1]:end[1], start[0]:end[0], :]
        return eye


def draw_point(image, iris_landmarks):
    for i in range(iris_landmarks.shape[0]):
        start = (int(iris_landmarks[i,0] * image.shape[1]),
                 int(iris_landmarks[i,1] * image.shape[0]))
        cv2.circle(image, start, 2, (255,0,0), -1)
    return image
#--------------
def detect_iris(eye_frame, is_right_eye=False):
    side_low = 64
    eye_frame_low = cv2.resize(
        eye_frame, (side_low, side_low), interpolation=cv2.INTER_AREA
    )

    model_path = "models/iris_landmark.tflite"

    if is_right_eye:
        eye_frame_low = np.fliplr(eye_frame_low)

    outputs = tflite_inference(eye_frame_low / 127.5 - 1.0, model_path)
    eye_contours_low = np.reshape(outputs[0], (71, 3))
    iris_landmarks_low = np.reshape(outputs[1], (5, 3))

    eye_contours = eye_contours_low / side_low
    iris_landmarks = iris_landmarks_low / side_low

    #st.write(iris_landmarks)

    if is_right_eye:
        eye_contours[:, 0] = 1 - eye_contours[:, 0]
        iris_landmarks[:, 0] = 1 - iris_landmarks[:, 0]

    return eye_contours, iris_landmarks

def tflite_inference(inputs, model_path, dtype=np.float32):

    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    for inp, inp_det in zip(inputs, input_details):
        interpreter.set_tensor(inp_det["index"], np.array(inp[None, ...], dtype=dtype))

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    outputs = [interpreter.get_tensor(out["index"]) for out in output_details]

    return outputs
#----------

if __name__ == "__main__":

    img_file_buffer = st.camera_input("Take a picture")

    # covert image to numpy
    if img_file_buffer is not None:
        # To read image file buffer with OpenCV:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # cut the eye and feed it to the model
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:

            # for idx, (frame, frame_rgb) in enumerate(source):
            frame_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = np.array(
                 [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                #landmarks = landmarks.T

            #st.write("shape=", cv2_img.shape[0], cv2_img.shape[1] )

            # initalize the model
            with tf.Session() as sess:
                # load best model
                logger = Logger('INC', '3A4Bh-Ref25', "", config, dir="models/")
                model = load_model(sess, 'INC', '3A4Bh-Ref25', logger)

                left_eye = cut_image(cv2_img, 70, 114)
                left_eye_copy = left_eye.copy()
                left_eye = run_me(left_eye, model, sess)


                right_eye = cut_image(cv2_img, 285, 345)
                right_eye_copy = right_eye.copy()
                right_eye = run_me(right_eye, model, sess)


                st.image(cv2_img)



            # draw point on the big picture

            #st.image(cv2_img)
            #st.write("shape=",left_eye_copy.shape)
            eye_contours, iris_landmarks = detect_iris(left_eye_copy, is_right_eye=False)
            left_eye_old = draw_point(left_eye_copy, iris_landmarks)

            eye_contours, iris_landmarks = detect_iris(right_eye_copy, is_right_eye=True)
            right_eye_old = draw_point(right_eye_copy, iris_landmarks)



            column1, column2 = st.columns(2)
            with column1:
                st.write("Left eye")
                st.write("New Model")
                st.image(left_eye, width=200)
                st.write("Old Model")
                st.image(left_eye_old, width=200)

            with column2:
                st.write("Right eye")
                st.write("New Model")
                st.image(right_eye, width=200)
                st.write("Old Model")
                st.image(right_eye_old, width=200)

    #run_me('INC', '3A4Bh-Ref25', logger)