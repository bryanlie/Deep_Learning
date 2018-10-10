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

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)
p_left_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_right), num_samples=1)

y = 1.0 - tf.to_float(action)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_vars = optimizer.compute_gradients(xentropy)
grads = [grad for grad, var in grads_vars]
grad_placeholder = []
grads_vars_feed = []
for grad, var in grads_vars:
    grad_ph = tf.placeholder(tf.float32, shape=grad.get_shape())
    grad_placeholder.append(grad_ph)
    grads_vars_feed.append((grad_ph, var))
training_op = optimizer.apply_gradients(grads_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

env = gym.make('CartPole-v0')

n_games_per_update = 10
n_max_steps = 1000
n_iters = 250
save_iters = 10
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iters):
        print("\rIteration: {}".format(iteration), end='')
        all_rewards = []
        all_grads = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, grads_val = sess.run([action, grads], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(grads_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_grads.append(current_gradients)

        all_rewards = discount_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_idx, grad_ph in enumerate(grad_placeholder):
            mean_grads = np.mean([reward * all_grads[game_idx][step][var_idx]
                                  for game_idx, rewards in enumerate(all_rewards)
                                      for step, reward in enumerate(rewards)], axis=0)
            feed_dict[grad_ph] = mean_grads
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iters == 0:
            saver.save(sess, './policy_net_pg.ckpt')

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


frames = render_policy_net("./policy_net_pg.ckpt", action, X, n_max_steps=1000)
video = plot_animation(frames)
plt.show()
