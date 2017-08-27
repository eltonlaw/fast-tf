""" Build and export TensorFlow computational graphs """
import os
import logging
# Quiets TensorFlow compile errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pylint: disable=wrong-import-position
import tensorflow as tf


# pylint: disable=invalid-name
def model(FLAGS):
    """ Test Experiment

    Parameters
    ----------
    FLAGS: Dictionary of model parameters

    Returns
    -------
    TensorFlow MetaGraphDef proto
    """
    logger = logging.getLogger(__name__)
    tf.set_random_seed(FLAGS.seed)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])


    # Initialize variables in parameter servers with greedy strategy
    # ps = FLAGS.ps_hosts.split(",")
    # # workers = FLAGS.worker_hosts.split(",")
    # load_fn = tf.contrib.training.byte_size_load_fn
    # greedy = tf.contrib.training.GreedyLoadBalancingStrategy(len(ps),
    #                                                          load_fn)
    # with tf.device(tf.train.replica_device_setter(ps_tasks=len(ps),
    #                                               ps_strategy=greedy)):
    # partitioner kwarg breaks up large variables into smaller ones
    W = tf.get_variable("weight", shape=[784, 10])
    #                    partitioner=tf.fixed_size_partitioner(3))
    b = tf.get_variable("bias", shape=[10])

    y_ = tf.matmul(x, W) + b

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

    train = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    score = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("score", score)

    merged_summary = tf.summary.merge_all()

    tf.add_to_collection('inputs', x)
    tf.add_to_collection('inputs', y)
    tf.add_to_collection('train_op', train)
    tf.add_to_collection('score_op', score)
    tf.add_to_collection('summary_op', merged_summary)
    collection_list = tf.get_default_graph().get_all_collection_keys()

    # g.finalize()
    meta_graph = tf.train.export_meta_graph(
        filename=os.path.join(FLAGS.log_dir, "meta_graph_def"),
        as_text=True,
        collection_list=collection_list
    )
    logger.debug("`test_experiment` MetaGraph Built")
    return meta_graph
