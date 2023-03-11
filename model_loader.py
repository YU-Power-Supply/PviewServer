import tensorflow as tf

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)

        return self.model
