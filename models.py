""" Build and export TensorFlow computational graphs """
import os
# Quiets TensorFlow compile errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
import tensorflow as tf


# pylint: disable=invalid-name
def test_experiment(FLAGS):
    """ Test Experiment

    PARAMETERS
    ----------
    FLAGS: Dictionary of model parameters

    RETURNS
    -------
    TensorFlow MetaGraphDef proto
    """
    tf.set_random_seed(FLAGS.seed)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("score", score)

    merged_summary = tf.summary.merge_all()

    tf.add_to_collection('x', x)
    tf.add_to_collection('y_', y_)
    tf.add_to_collection('train_op', train)
    tf.add_to_collection('score_op', score)
    tf.add_to_collection('summary_op', merged_summary)
    collection_list = tf.get_default_graph().get_all_collection_keys()

    # g.finalize()
    meta_graph = tf.train.export_meta_graph(
        collection_list=collection_list
    )
    return meta_graph
