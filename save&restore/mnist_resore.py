import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import pylab

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./log/521model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./log'))  
    
    graph = tf.get_default_graph ()
    x = graph.get_tensor_by_name ("x_input:0")
    
    op_to_restore = graph.get_tensor_by_name ("op_restore:0")
    
    
    batch_xs, batch_ys = mnist.train.next_batch(2)
    predv = sess.run(op_to_restore, feed_dict={x: batch_xs})
    
    print(predv,batch_ys)
    
    im = batch_xs[0]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1,28)
    pylab.imshow(im)
    pylab.show()