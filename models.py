import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def experiment(ps):
    mnist = input_data.read_data_sets('../../../MNIST_data', one_hot=True)
    tf.set_random_seed(ps.seed)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_op = tf.train.GradientDescentOptimizer(ps.lr).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(ps.epochs):
            batch = mnist.train.next_batch(ps.batch_size)
            _, s, a = sess.run([train_op, merged_summary, accuracy],
                               feed_dict={x: batch[0], y_: batch[1]})
            if epoch % ps.save_step == 0:
                print("Epoch {} Training Error = {}".format(epoch+1, a))

        test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images,
                                                 y_: mnist.test.labels})
    return test_accuracy
