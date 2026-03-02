import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

#DATA CREATION

num_samples_per_class = 1000
num = num_samples_per_class *2

##INFINITY##
#inner dot on the left
mu, sigma = 0, .6
s00 = np.random.normal(mu, sigma, int(num_samples_per_class/2))
theta0 = np.random.uniform(0,2*np.pi,int(num_samples_per_class/2))
C00_samples = np.array([s00 * np.cos(theta0)-4, s00*np.sin(theta0)])
C00_samples = np.transpose(C00_samples)
print(C00_samples.shape)

#inner dot on the right
s01 = np.random.normal(mu, sigma, int(num_samples_per_class/2))
theta0 = np.random.uniform(0,2*np.pi,int(num_samples_per_class/2))
C01_samples = np.array([s01 * np.cos(theta0)+4, s01*np.sin(theta0)])
C01_samples = np.transpose(C01_samples)
print(C01_samples.shape)

#make them into one set
C0_samples = np.vstack([C00_samples, C01_samples]);

#outer ring on the left
mu, sigma = 3, .5
s10 = np.random.normal(mu, sigma, int(num_samples_per_class/2)) + .05
theta1 = np.random.uniform(0,2*np.pi,int(num_samples_per_class/2))
C10_samples = np.array([s10 * np.cos(theta1)-4, s10*np.sin(theta1)])
C10_samples = np.transpose(C10_samples)
print(C10_samples.shape)

#outer ring on the right
s11 = np.random.normal(mu, sigma, int(num_samples_per_class/2)) + .05
theta1 = np.random.uniform(0,2*np.pi,int(num_samples_per_class/2))
C11_samples = np.array([s11 * np.cos(theta1)+4, s11*np.sin(theta1)])
C11_samples = np.transpose(C11_samples)
print(C10_samples.shape)

#make them into one set
C1_samples = np.vstack([C10_samples, C11_samples]);

# Testing data
s0 = np.random.normal(mu, sigma, num_samples_per_class)
theta0 = np.random.uniform(0,2*np.pi,num_samples_per_class)
T0_samples = np.array([s0 * np.cos(theta0) + 4, s0*np.sin(theta0) + 5])
T0_samples = np.transpose(T0_samples)
print(T0_samples.shape)

s1 = np.random.normal(mu, sigma, num_samples_per_class) + 4
theta1 = np.random.uniform(0,2*np.pi,num_samples_per_class)
T1_samples = np.array([s1 * np.cos(theta1) + 4, s1*np.sin(theta1) + 5])
T1_samples = np.transpose(T1_samples)
print(T1_samples.shape)

inputs = np.vstack((C0_samples, C1_samples)).astype(np.float32)
Tinputs = np.vstack ((T0_samples, T1_samples)).astype(np.float32)

targets = np.vstack((
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)

Ttargets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))
Ttargets = np.reshape(Ttargets,[num,1])
print(Tinputs.shape, Ttargets.shape)


#linear classifier variables
input_dim = 2
output_dim = 1

W1 = tf.Variable(tf.random.normal([2,16], dtype = tf.float32)) #2 inputs to 16 hidden
b1 = tf.Variable(tf.zeros([16], dtype=tf.float32))
W2 = tf.Variable(tf.random.normal([16,1], dtype = tf.float32)) #16 hidden to 1 out
b2 = tf.Variable(tf.zeros([1], dtype=tf.float32))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model = tf.keras.models.Sequential([
    layers.Dense(16, input_shape=(2,), activation='relu' ),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(inputs, targets, epochs=50, verbose=0)

predictions = model(inputs)
predictions_prob = tf.sigmoid(predictions).numpy()

# Create a grid of points covering the data space
x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Run every grid point through the model
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
grid_preds = tf.sigmoid(model(grid_points)).numpy().reshape(xx.shape)

# Plot the decision boundary as a background color
plt.contourf(xx, yy, grid_preds, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
plt.contour(xx, yy, grid_preds, levels=[0.5], colors='black')  # the boundary line

# Plot the actual data points on top
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions_prob > 0.5, cmap='bwr', edgecolors='k', s=10)
plt.title("Decision Boundry")
plt.show()




