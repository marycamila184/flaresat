import tensorflow as tf
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable(package='flaresat')
class F1Score(tf.keras.metrics.Metric):
    """Pixel-level F1 with TP/FP/FN accumulated over all batches of the epoch.

    Same definition as the evaluation (train/evaluation_cross_train.py). A per-image F1 would score every
    empty-mask patch (wildfire, urban) as 0 whatever the prediction, hiding their false positives.
    """

    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_variable(shape=(), initializer='zeros', name='tp')
        self.fp = self.add_variable(shape=(), initializer='zeros', name='fp')
        self.fn = self.add_variable(shape=(), initializer='zeros', name='fn')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0.5, 'float32')
        y_pred = tf.cast(y_pred > self.threshold, 'float32')
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        return 2 * precision * recall / (precision + recall + K.epsilon())

    def get_config(self):
        return {**super().get_config(), 'threshold': self.threshold}
