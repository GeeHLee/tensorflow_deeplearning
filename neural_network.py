from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# parameters:
learning_rate = 0.01
num_steps = 400
batch_size = 128
display_step = 100

# network parameters:
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784  # img shape = 28*28
num_classes = 10  # 0-9 digital number

# tf Graph input:
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# layers weights and bias
weights = {'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
           'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))}
bias = {'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))}


# create model:
def neural_net(x):
    # hidden connection layer with 256 neurons.
    layer_1 = tf.add(tf.matmul(x, weights['h1']), bias['b1'])
    # hidden connection layer with 256 neurons.
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), bias['b2'])
    # output layer
    out_layer = tf.matmul(layer_2, weights['out']) + bias['out']
    return out_layer


# Cons model
logits = neural_net(X)

# loss function:
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer:
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

# evaluation model:
corrected_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))

# initialization
init = tf.global_variables_initializer()

# start training
with tf.Session() as sess:
    # run initialization
    sess.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # optimization op
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # batch loss and accuracy:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss={:.4f}".format(loss) + \
                  ", Training Accuracy={:.3f}".format(acc))

    print("Optimization Finished")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

