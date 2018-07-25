# fast_neural_style_transfer
This code refers to https://github.com/hzy46/fast-neural-style-tensorflow. <br>
You can see the middle result in the output/ <br>
And you can see the finally result in the generated/ <br>
Last version exists the Checkerboard Artifacts because I use the tf.nn.conv2d_transpose <br>
And you can see the https://distill.pub/2016/deconv-checkerboard/ to understand about Checkerboard Artifacts<br>
It works still not good,and I think because it just train the three images for 1000 ecophs <br>
<br>
Train data 5000 steps about 20000 images the result become better<br>
![image](https://github.com/MeetUo/fast_neural_style_transfer/blob/master/output/0.jpg) 


