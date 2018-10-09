import numpy as np
import tensorflow as tf
from functools import partial


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

bn_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int32, shape=(None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')


with tf.name_scope('dnn'):
    he_init = tf.variance_scaling_initializer()

    bn_layer = partial(tf.layers.batch_normalization, training=training, momentum=bn_momentum)

    dense_layer = partial(tf.layers.dense, kernel_initializer=he_init)

    hidden1 = dense_layer(X, n_hidden1, name='hidden1')
    bn1 = tf.nn.elu(bn_layer(hidden1))

    hidden2 = dense_layer(bn1, n_hidden2, name='hidden2')
    bn2 = tf.nn.elu(bn_layer(hidden2))

    logits_bfbn = dense_layer(bn2, n_outputs, name='outputs')
    logits = bn_layer(logits_bfbn)


with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 40
batch_size = 200

extra_updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_updates_ops], feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val, loss_val = sess.run([accuracy, loss], feed_dict={X: X_test, y: y_test})
        print("Epoch: ", epoch, "\taccuracy: ", accuracy_val, "\tloss: ", loss_val)




