import tensorflow as tf

mnist = tf.keras.datasets.mnist # dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data() # data split
x_train, x_test = x_train / 255.0, x_test / 255.0 # data normalization

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),# flatten
  tf.keras.layers.Dense(128, activation='relu'), # full connected layer,  28x28 -> 128
  tf.keras.layers.Dropout(0.2), # fixed dropout
  tf.keras.layers.Dense(10) # 10 different classes, from 0 to 9
]) # model configuration

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy() # logits -> probability

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss function

# loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy']) # optimization configuration

model.fit(x_train, y_train, epochs=5) # training configuration

model.evaluate(x_test,  y_test, verbose=2) # evaluation configuration

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
]) # Encapsulation
probability_model(x_test[:5])