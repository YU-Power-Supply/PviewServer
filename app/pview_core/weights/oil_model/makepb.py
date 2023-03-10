import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('PviewServer/app/pview_core/weights/oil_model/oilly_model_weight_1203.h5')

# Convert the model to the SavedModel format
tf.saved_model.save(model, 'PviewServer/app/pview_core/weights/oil_model/1/')