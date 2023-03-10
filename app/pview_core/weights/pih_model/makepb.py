import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('PviewServer/app/pview_core/weights/pih_model/pih_model_weight_0912.h5')

# Convert the model to the SavedModel format
tf.saved_model.save(model, 'PviewServer/app/pview_core/weights/pih_model/1/')