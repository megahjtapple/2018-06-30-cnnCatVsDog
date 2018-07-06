import os
import numpy as np
import tensorflow as tf
import input_data
import model
import proj_constants

from PIL import Image
import matplotlib.pyplot as plt

def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''

    # you need to change the directories to yours.
    train_dir = proj_constants.train_dir
    train, train_label = input_data.get_files(train_dir)
    #image_array = get_one_image(train)

    img_dir = "/home/user000/Desktop/testImg.jpg"
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image_array = np.array(image)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = proj_constants.classes

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # you need to change the directories to yours.
        logs_train_dir = proj_constants.logs_train_dir

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            print(prediction)
            max_index = np.argmax(prediction)
            prediction_final = ""
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
                prediction_final = "cat"
            elif max_index==1:
                print('This is a dog with possibility %.6f' %prediction[:, 1])
                prediction_final = "dog"
            else:
                print('This is a flower with possibility %.6f' %prediction[:, 1])
                prediction_final = "flower"
            print("\n\nThis is " + prediction_final)
            plt.imshow(image_array)
            plt.show()





