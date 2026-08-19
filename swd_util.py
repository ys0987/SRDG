import tensorflow as tf
import numpy as np


def sw_loss(true_distribution, generated_distribution,num_projections,batch_size):

    s = true_distribution.get_shape().as_list()[-1]

    # num_projections=140
    # batch_size=140
    theta = tf.random.normal(shape=[s, num_projections])
    theta = tf.nn.l2_normalize(theta, axis=0)

    # project the samples (images). After being transposed, we have tensors
    # of the format: [projected_image1, projected_image2, ...].
    # Each row has the projections along one direction. This makes it
    # easier for the sorting that follows.
    projected_true = tf.transpose(
    tf.matmul(true_distribution, theta))

    projected_fake = tf.transpose(
    tf.matmul(generated_distribution, theta))

    sorted_true, true_indices = tf.nn.top_k(
    projected_true,
    batch_size)

    sorted_fake, fake_indices = tf.nn.top_k(
    projected_fake,
    batch_size)
    # print(sorted_fake.shape, fake_indices.shape)

    # For faster gradient computation, we do not use sorted_fake to compute
    # loss. Instead we re-order the sorted_true so that the samples from the
    # true distribution go to the correct sample from the fake distribution.
    # This is because Tensorflow did not have a GPU op for rearranging the
    # gradients at the time of writing this code.

    # It is less expensive (memory-wise) to rearrange arrays in TF.
    # Flatten the sorted_true from [batch_size, num_projections].
    flat_true = tf.reshape(sorted_true, [-1])

    # Modify the indices to reflect this transition to an array.
    # new index = row + index
    rows = np.asarray(
    [batch_size * np.floor(i * 1.0 / batch_size)
    for i in range(num_projections * batch_size)])
    rows = rows.astype(np.int32)
    flat_idx = tf.reshape(fake_indices, [-1, 1]) + np.reshape(rows, [-1, 1])

    # The scatter operation takes care of reshaping to the rearranged matrix
    shape = tf.constant([batch_size * num_projections])
    rearranged_true = tf.reshape(
    tf.scatter_nd(flat_idx, flat_true, shape),
    [num_projections, batch_size])

    return tf.reduce_mean(tf.square(projected_fake - rearranged_true))

def sw_loss2(true_distribution, generated_distribution, num_projections=64):
    """
    true_distribution: [B, D]
    generated_distribution: [B, D]
    """
    true_distribution = tf.convert_to_tensor(true_distribution, dtype=tf.float32)
    generated_distribution = tf.convert_to_tensor(generated_distribution, dtype=tf.float32)

    # Ensure 2D [B, D]
    true_distribution = tf.reshape(true_distribution, [tf.shape(true_distribution)[0], -1])
    generated_distribution = tf.reshape(generated_distribution, [tf.shape(generated_distribution)[0], -1])

    d = tf.shape(true_distribution)[1]  # D (e.g., 3072)

    # Random projection directions: [D, P]
    theta = tf.random.normal(shape=tf.stack([d, num_projections]))
    theta = tf.nn.l2_normalize(theta, axis=0)

    # Project: [B, P] -> transpose to [P, B]
    projected_true = tf.transpose(tf.matmul(true_distribution, theta))
    projected_fake = tf.transpose(tf.matmul(generated_distribution, theta))

    # Sort along batch axis (last axis): [P, B]
    sorted_true = tf.sort(projected_true, axis=-1)
    sorted_fake = tf.sort(projected_fake, axis=-1)

    return tf.reduce_mean(tf.square(sorted_true - sorted_fake))
