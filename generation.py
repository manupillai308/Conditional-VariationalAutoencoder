import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


with tf.Session() as sess:
	tf.saved_model.loader.load(sess,["serve"],"./saved_model/decoder")
	coding = sess.graph.get_tensor_by_name("concat:0")
	output = sess.graph.get_tensor_by_name("Sigmoid:0")
	for j in range(50):
		for i in range(10):	
			plt.subplot(2,5,i+1)
			code = np.concatenate([np.random.normal(size=[1, 10]), np.eye(10)[i].reshape((1,10))], 1)
			generated_image = sess.run(output, feed_dict={coding:code})
			plt.imshow(generated_image.reshape((28,28)), cmap="gray")
			plt.axis("off")
		plt.savefig("./images/%d.jpg" % j)
		plt.show(block=False)
		plt.pause(0.001)
	
