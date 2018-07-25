from process_image import *
import losses
import vgg_v19
import my_model
import time
import imageio

class CONFIG:
    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256
    BATCH_SIZE = 4
    COLOR_CHANNELS = 3
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat'  # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    STYLE_IMAGE = 'images/wave.jpg'  # Style image to use.
    CONTENT_IMAGE_FILE = 'E:/python学习/PycharmProjects/train2014/train2014'  # Content image to use.
    OUTPUT_DIR = 'output/'

    W_CONTENT = 1
    W_STYLE = 255

def train_fnst():
    with tf.Graph().as_default():
        style_feature = losses.get_style_feature(CONFIG.STYLE_IMAGE,
                                                 CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)
        with tf.Session() as sess:
            train_data = train_image(CONFIG.BATCH_SIZE, CONFIG.IMAGE_HEIGHT,
                               CONFIG.IMAGE_WIDTH, CONFIG.CONTENT_IMAGE_FILE,epochs=2)
            y_data = my_model.get_my_model(train_data)
            process_data = [reshape_and_normalize_image(image)
                          for image in tf.unstack(y_data, axis=0, num=CONFIG.BATCH_SIZE)]
            process_data = tf.stack(process_data)
            layer_feature = vgg_v19.load_vgg_model(CONFIG.VGG_MODEL, CONFIG.IMAGE_HEIGHT,
                                               CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS, tf.concat([process_data, train_data], 0));
            style_loss = losses.get_style_loss(style_feature,layer_feature)
            content_loss = losses.get_content_loss(layer_feature)
            total_loss = CONFIG.W_STYLE*style_loss+CONFIG.W_CONTENT*content_loss
            global_step = tf.Variable(0, name="global_step", trainable=False)

            variable_to_train = []
            for variable in tf.trainable_variables():
                if not (variable.name.startswith('vgg')):
                    variable_to_train.append(variable)
            train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step, var_list=variable_to_train)
            print(variable_to_train)

            variables_to_restore = []
            for v in tf.global_variables():
                if not (v.name.startswith('vgg')):
                    variables_to_restore.append(v)
            saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)

            #开始训练
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while not coord.should_stop():
                    start_time = time.time()
                    _, loss_t, step ,y= sess.run([train_op,total_loss, global_step,y_data])
                    elapsed_time = time.time() - start_time
                    """logging"""
                    # print(step)
                    if step%10 == 0:
                      print('step: %d, total Loss %f, secs/step: %f'
                                        %(step,loss_t, elapsed_time))
                    if step % 5000 == 0:
                        for index in range(3):
                            imge = np.clip(y[index],0,255).astype('uint8')
                            imageio.imsave(CONFIG.OUTPUT_DIR+str(index)+".jpg",imge);
                        saver.save(sess, './output/fast-style-model.ckpt-done')
            except tf.errors.OutOfRangeError:
                print("done")
                saver.save(sess, './output/fast-style-model.ckpt-done')
            finally:
                coord.request_stop()
            coord.join(threads)