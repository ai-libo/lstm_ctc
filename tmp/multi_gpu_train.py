import os.path
import re
import time
import numpy as np
import tensorflow as tf
#import cifar10
import pdb
batch_size=128
#train_dir='/tmp/cifar10_train'
max_steps=1000000
num_gpus=2
#log_device_placement=False
def tower_loss(scope):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    Returns:
         Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()
    logits = cifar10.inference(images)
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = cifar10.loss(logits, labels)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss
def average_gradients(tower_grads):
    """
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      
        global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
        print (global_step)
        sess = tf.Session()
        sess.run(global_step)
        num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                                 batch_size)
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                                                        global_step,
                                                                        decay_steps,
                                                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                                                        staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)
        tower_grads = []
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                    loss = tower_loss(scope)
                    tf.get_variable_scope().reuse_variables()
                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization(同步) point across all towers.
        grads = average_gradients(tower_grads)
        summaries.append(tf.scalar_summary('learning_rate', lr))
        for grad, var in grads:
            if grad is not None:
                summaries.append(
                    tf.histogram_summary(var.op.name + '/gradients', grad))

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            summaries.append(tf.histogram_summary(var.op.name, var))

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_summary(summaries)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
        for step in range(max_steps):
            start_time = time.time()
            _, loss_value = sess.run([apply_gradient_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_examples_per_step = batch_size * num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / num_gpus
                format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                            'sec/batch)')
                print (format_str % (step, loss_value,
                                                         examples_per_sec, sec_per_batch))
                if step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
            if step % 1000 == 0 or (step + 1) == max_steps:
                saver.save(sess, '/tmp/cifar10_train/model.ckpt', global_step=step)
#cifar10.maybe_download_and_extract()
train()