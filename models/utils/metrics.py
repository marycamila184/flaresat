import tensorflow as tf
from tensorflow.keras import backend as K

def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=[1, 2, 3])
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=[1, 2, 3])
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=[1, 2, 3])

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)
