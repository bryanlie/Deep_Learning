import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


reset_graph()

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label='Positive')
plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label='Negative')
plt.legend()
plt.show()

X_moons_wbias = np.c_[np.ones((m, 1)), X_moons]

y_moons = y_moons.reshape(-1, 1)

test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_wbias[: -test_size]
X_test = X_moons_wbias[-test_size:]
y_train = y_moons[:-test_size]
y_test = y_moons[-test_size:]


def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch


n_inputs = 2

X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name='theta')
logits = tf.matmul(X, theta, name='logits')
y_proba = tf.sigmoid(logits)
loss = tf.losses.log_loss(y, y_proba)

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()

n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index  in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch: ", epoch, "\tloss: ", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    y_pred = (y_proba_val >= 0.5)

    print("Precision score: ", precision_score(y_test, y_pred))
    print("Recall score: ", recall_score(y_test, y_pred))


X_train_enhanced = np.c_[X_train, np.square(X_train[:, 1]), np.square(X_train[:, 2]),
                         X_train[:, 1] ** 3, X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test, np.square(X_test[:, 1]), np.square(X_test[:, 2]),
                         X_test[:, 1] ** 3, X_test[:, 2] ** 3]


def log_reg(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs_wbias = int(X.get_shape()[1])
    with tf.name_scope('log_reg'):
        with tf.name_scope('model'):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs_wbias, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name='theta')
            logits = tf.matmul(X, theta, name='logits')
            y_proba = tf.sigmoid(logits)
        with tf.name_scope('train'):
            loss = tf.losses.log_loss(y, y_proba, scope='loss')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
        with tf.name_scope('init'):
            init = tf.global_variables_initializer()

    return y_proba, loss, training_op, loss_summary, init

n_inputs = 6

X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

y_proba, loss, training_op, loss_summary, init = log_reg(X, y)

n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    start_epoch = 0
    sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
        if epoch % 500 == 0:
            print("Epoch: ", epoch, "\tloss: ", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
    y_pred = (y_proba_val >=0.5)
    print("Precision score: ", precision_score(y_test, y_pred))
    print("Recall score: ", recall_score(y_test, y_pred))


