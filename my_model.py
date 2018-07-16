import  tensorflow as tf


def get_my_model(image):
    MODEL_LAYERS = [
        'conv1','conv2','conv3',
        'res1','res2','res3','res4','res5',
        'dconv1','dconv2','dconv3']
    model_arg =[
        (3,32,9,1),(32,64,3,2),(64,128,3,2),
        (128,3,1,0),(128,3,1,0),(128,3,1,0),(128,3,1,0),(128,3,1,0),
        (128,64,3,2),(64,32,3,2),(32,3,9,1)
    ]

    def instance_norm(x):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

    def relu(layer):
        return tf.nn.relu(layer)

    def tanh(layer):
        return tf.nn.tanh(layer)

    def conv2d(x,input_filter,output_filter, kernal, strides):
        with tf.variable_scope('conv2d'):
            shape=[kernal,kernal,input_filter,output_filter]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            return tf.nn.conv2d(x,filter = weight,strides=[1,strides,strides,1],
                                padding='SAME',name='conv')

    def dconv2d(x,input_filter,output_filter,kernal,strides,out_shape):
        with tf.variable_scope('dconv2d'):
            shape = [kernal,kernal,output_filter,input_filter]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            return tf.nn.conv2d_transpose(x,filter = weight,strides=[1,strides,strides,1],
                                          padding='SAME',name='dconv',output_shape=out_shape)
    def reslayer(x,filter,kernel,strides):
        with tf.variable_scope('resnet'):
            conv1 = conv2d(x, filter, filter, kernel, strides)
            conv2 = conv2d(relu(conv1), filter, filter, kernel, strides)
            residual = x + conv2
            return residual

    img = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    conv = img;
    out_shpaes = [];
    dnum = 0;
    for name,arg_num in zip(MODEL_LAYERS,model_arg):
        if name.startswith('c'):
            with tf.variable_scope(name):
                out_shpaes.append(tf.shape(conv));
                conv = relu(instance_norm(conv2d(conv,arg_num[0],arg_num[1],arg_num[2],arg_num[3])))
        elif name.startswith('r'):
            with tf.variable_scope(name):
                conv = relu(instance_norm(reslayer(conv,arg_num[0],arg_num[1],arg_num[2])))
        else:
            with tf.variable_scope(name):
                dnum += 1;
                out_shape = out_shpaes[len(out_shpaes)-dnum]
                if name.endswith('3'):
                    conv = tanh(instance_norm(dconv2d(conv, arg_num[0], arg_num[1], arg_num[2], arg_num[3],out_shape)))
                else:
                    conv = relu(instance_norm(dconv2d(conv, arg_num[0], arg_num[1], arg_num[2], arg_num[3],out_shape)))

    y = (conv + 1) * 127.5
    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

    return y