""" All the boilerplate to run a graph"""
# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=too-many-locals
# pylint: disable=logging-format-interpolation
import os
import time
import argparse
import logging
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from models import test_experiment

def run_session(mgd, data):
    """ Runs a TensorFlow MetaGraphDef in a session

    Parameters
    ----------
    mgd: TensorFlow MetaGraphDef

    Returns
    -------
    Results printed to stdout (which will be caught by `run.sh`)
    """
    start_time = time.time()
    tf.reset_default_graph()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        new_saver = tf.train.import_meta_graph(mgd)

        x = tf.get_collection("inputs")[0]
        y_ = tf.get_collection("inputs")[1]
        train_op = tf.get_collection("train_op")[0]
        summary_op = tf.get_collection("summary_op")[0]
        score_op = tf.get_collection("score_op")[0]
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        save_path = os.path.join(FLAGS.log_dir, "experiment")

        for epoch_i in range(FLAGS.epochs):
            for batch_i in range(FLAGS.batches_per_epoch):
                batch = data.train.next_batch(FLAGS.batch_size)
                # pylint: disable=unused-variable
                _, u, c = sess.run([train_op, summary_op, score_op],
                                   feed_dict={x: batch[0], y_: batch[1]})
                writer.add_summary(u, epoch_i * batch_i)
            if epoch_i % FLAGS.save_step == 0:
                print("E{} Training Error = {}".format(epoch_i,
                                                       c))
                saver.save(sess, save_path)

        final_train_score = c
        test_score = score_op.eval(feed_dict={x: data.test.images,
                                              y_: data.test.labels})
        run_time = time.strftime('%dD%HH:%MM:%SS',
                                 time.gmtime(time.time() - start_time))
        results = {
            "final_train_score": final_train_score,
            "test_score": test_score,
            "run_time":run_time
        }
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default="./logs/")
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--train_dir', type=str, default="./data/")
    parser.add_argument('--ps_hosts', type=str, default="localhost:2222",
                        help="Comma-separated list of hostname:port pairs")
    parser.add_argument('--worker_hosts', type=str,
                        default="localhost:2223,localhost:2224",
                        help="Comma-separated list of hostname:port pairs")
    # Grab command line args
    FLAGS = parser.parse_args()

    # Setup logging
    logging.basicConfig(filename="{}/test.log".format(FLAGS.log_dir),
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    tf.set_random_seed(FLAGS.seed)
    # Load Data
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
    logger.debug("Data Loaded")
    # Compute total number of batches in one training epoch
    FLAGS.batches_per_epoch = mnist.train.images.shape[0]//FLAGS.batch_size
    # Build Graph
    meta_graph_def = test_experiment(FLAGS)
    # Feed data into graph and run graph in session
    result = run_session(meta_graph_def, mnist)

    # Save params and result together (OFFLOADED TO BASH)
    # with open(os.path.join(FLAGS.log_dir, "results.txt"), "a") as f:
    #    f.write(str(vars(FLAGS)))
    #    f.write("\n")
    #    f.write(str(result))
    #    f.write("\n \n")
    logger.info("Parameters:\n{}".format(str(vars(FLAGS))))
    logger.info("Results:\n{}".format(str(result)))
