from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('steps', 100, 'The number of steps to train a model')
tf.app.flags.DEFINE_integer('hidden_layers', 2, 'The number of hidden layers in the neural network')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'The learning rate for the Adam optimisation algorithm')
tf.app.flags.DEFINE_string('model_dir', './model/', 'Directory where model parameters, graph, checkpoints, etc are saved')
tf.app.flags.DEFINE_string('export_dir', './export/', 'Directory where to export a model for TF serving')
FLAGS = tf.app.flags.FLAGS

INPUT_FEATURE = 'image'

def layer(inputs, filters, name):
    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[5, 5],
        padding="same",
        activation=None,
        name=name)
    batch_norm = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(batch_norm)
    pool = tf.layers.max_pooling2d(inputs=relu, pool_size=[2, 2], strides=2)
    return pool

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    current_layer = features[INPUT_FEATURE]

    for layer_id in range(FLAGS.hidden_layers):
        new_layer = layer(current_layer, 32, 'layer{}'.format(layer_id))
        current_layer = new_layer

    # Flatten tensor into a batch of vectors
    pool_flat = tf.layers.flatten(current_layer)

    # Dense Layer
    dense = tf.layers.dense(inputs=pool_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_receiver_fn():
    inputs = {
        INPUT_FEATURE: tf.placeholder(tf.float32, [None, 28, 28, 1]),
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main(_):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # reshape images
    # To have input as an image, we reshape images beforehand.
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    eval_data = eval_data.reshape(eval_data.shape[0], 28, 28, 1)

    # Create the Estimator
    training_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=20,
        save_checkpoints_steps=20)
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=FLAGS.model_dir,
        config=training_config)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={INPUT_FEATURE: train_data},
      y=train_labels,
      batch_size=FLAGS.steps,
      num_epochs=None,
      shuffle=True)
    train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=100,
      hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={INPUT_FEATURE: eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=100,
      start_delay_secs=0)

    # run !
    tf.estimator.train_and_evaluate(
      classifier,
      train_spec,
      eval_spec
    )

    # Save the model
    classifier.export_savedmodel(FLAGS.export_dir, serving_input_receiver_fn=serving_input_receiver_fn)

if __name__ == "__main__":
    tf.app.run()
