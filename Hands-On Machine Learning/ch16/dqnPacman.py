import gym
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as anim

env = gym.make('MsPacman-v0')
obs = env.reset()


mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2)
    img[img == mspacman_color] = 0
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)

img = preprocess_observation(obs)

input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n
initializer = tf.variance_scaling_initializer()


def q_network(X_state, name):
    prev_layer = X_state / 128.0
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(conv_n_maps, conv_kernel_sizes,
                                                                     conv_strides, conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size, strides=strides,
                                          padding=padding, activation=activation, kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
        return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
online_q_values, online_vars = q_network(X_state, name='q_networks/online')
target_q_values, target_vars = q_network(X_state, name='q_networks/target')

copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)


learning_rate = 0.001
momentum = 0.95

with tf.variable_scope('train'):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs), axis=1, keepdims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]


replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)


def sample_memories(batch_size):
    cols = [[], [], [], [], []]
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000


def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return np.argmax(q_values)

n_steps = 4000000
training_start = 10000
training_interval = 4
save_steps = 1000
copy_steps = 10000
discount_rate = 0.99
skip_start = 90
batch_size = 50
iteration = 0
checkpt_path = './dqn.ckpt'
done = True


loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0


with tf.Session() as sess:
    if os.path.isfile(checkpt_path + '.index'):
        saver.restore(sess, checkpt_path)
    else:
        init.run()
        copy_online_to_target.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        print("\rIteration {} \tTraining step {}/{} ({:.1f})% \tLoss {:5f} \t Mean Max-Q {:5f}   ".format(
            iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
        if done:
            obs = env.reset()
            for skip in range(skip_start):
                obs, reward, done, info = env.step(0)
                state = preprocess_observation(obs)

        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        total_max_q += q_values.max()
        game_length += 1
        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length - 0

        if iteration < training_start or iteration % training_interval != 0:
            continue

        X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(batch_size)
        next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        _, loss_val = sess.run([training_op, loss], feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

        if step % copy_steps == 0:
            copy_online_to_target.run()

        if step & save_steps == 0:
            saver.save(sess, checkpt_path)


frames = []
n_max_steps = 10000

with tf.Session() as sess:
    saver.restore(sess, checkpt_path)

    obs = env.reset()
    for step in range(n_max_steps):
        state = preprocess_observation(obs)

        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = np.argmax(q_values)

        obs, reward, done, info = env.step(action)

        img = env.render(mode='rgb_array')
        frames.append(img)

        if done:
            break


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch


def plot_animation(frames, repeat=False, interval=40):
    plt.close()
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return anim.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames),
                                   repeat=repeat, interval=interval)


video = plot_animation(frames)
plt.show()

env.close()

