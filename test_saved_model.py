import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., None]  # (10000, 28, 28, 1)

# Load saved model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Randomly select 10 test images
num_samples = 10
idx = np.random.choice(len(x_test), num_samples, replace=False)

images = x_test[idx]
labels = y_test[idx]

# Predict
preds = model.predict(images)
pred_labels = np.argmax(preds, axis=1)

# Plot results
plt.figure(figsize=(12, 3))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(images[i].reshape(28, 28), cmap="gray")
    plt.axis("off")
    plt.title(f"T:{labels[i]}\nP:{pred_labels[i]}")
plt.tight_layout()
plt.show()
