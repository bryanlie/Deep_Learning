import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as anim

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01


initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.float32, shape=[None, n_outputs])


hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)
# outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)

p_left_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_right), num_samples=1)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_env = 10
n_iter = 1000


envs = [gym.make('CartPole-v0') for _ in range(n_env)]
obss = [env.reset() for env in envs]


with tf.Session() as sess:
    init.run()
    for iteration in range(n_iter):
        target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in obss])
        action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(obss), y: target_probas})
        for env_idx, env in enumerate(envs):
            obs, reward, done, info = env.step(action_val[env_idx][0])
            obss[env_idx] = obs if not done else env.reset()
    saver.save(sess, "./policy_net_basic.ckpt")


for env in envs:
    env.close()


def render_policy_net(model_path, action, X, n_max_steps = 1000):
    frames = []
    env = gym.make("CartPole-v0")
    obs = env.reset()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            img = env.render(mode='rgb_array')
            frames.append(img)
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
    env.close()
    return frames


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


frames = render_policy_net("./policy_net_basic.ckpt", action, X)
video = plot_animation(frames)
plt.show()


