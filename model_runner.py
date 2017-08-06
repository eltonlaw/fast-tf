""" All the boilerplate to run a model"""
# import os
import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from models import test_experiment


def run_session(mgd):
    """ Runs a TensorFlow MetaGraphDef in a session

    Parameters
    ----------
    mgd: TensorFlow MetaGraphDef

    Returns
    -------
    Results printed to stdout (which will be caught by `run.sh`)

    """
    # pylint: disable=invalid-name, too-many-locals, unused-variable
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    tf.reset_default_graph()
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(mgd)

        x = tf.get_collection("inputs")[0]
        y_ = tf.get_collection("inputs")[1]
        train_op = tf.get_collection("train_op")[0]
        summary_op = tf.get_collection("summary_op")[0]
        score_op = tf.get_collection("score_op")[0]
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        save_path = os.path.join(FLAGS.log_dir, "experiment")

        for epoch in range(FLAGS.epochs):
            batch = mnist.train.next_batch(FLAGS.batch_size)
            # pylint: disable=unused-variable
            _, u, c = sess.run([train_op, summary_op, score_op],
                               feed_dict={x: batch[0], y_: batch[1]})
            if epoch % FLAGS.save_step == 0:
                print("Epoch {} Training Error = {}".format(epoch+1, c))
                saver.save(sess, save_path)

        final_train_score = c
        test_score = score_op.eval(feed_dict={x: mnist.test.images,
                                              y_: mnist.test.labels})
        results = {
            "final_train_score": final_train_score,
            "test_score": test_score
        }
    return results


if __name__ == "__main__":
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1000)
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
    FLAGS = parser.parse_args()

    meta_graph_def = test_experiment(FLAGS)
    result = run_session(meta_graph_def)

    # Save params and result together (OFFLOADED TO BASH)
    # with open(os.path.join(FLAGS.log_dir, "results.txt"), "a") as f:
    #    f.write(str(vars(FLAGS)))
    #    f.write("\n")
    #    f.write(str(result))
    #    f.write("\n \n")
    print(str(vars(FLAGS)))
    print(result)
