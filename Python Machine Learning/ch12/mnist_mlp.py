import numpy as np
import matplotlib.pyplot as plt

mnist = np.load('data/mnist_scaled.npz')
X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]

print(X_train.shape)


from neuralnet import NeuralNetMLP

n_epochs = 200
nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=n_epochs, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)
nn.fit(X_train=X_train[:55000], y_train=y_train[:55000], X_valid=X_train[55000:], y_valid=y_train[55000:])

plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.xlabel('Epochs')
plt.ylabel('Cost')

plt.figure()
plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])

print('Test accuracy: %.2f%%' % (acc * 100))

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True)
ax = ax.ravel()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

ax[0].set_xticks([])
ax[0].set_yticks([])


plt.show()

