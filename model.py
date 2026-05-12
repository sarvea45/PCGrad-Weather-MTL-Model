import tensorflow as tf

def build_mtl_model(input_shape=(7, 3)):
    """Builds a 1D-CNN multi-task model for spatio-temporal data."""
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')
    
    # Shared 1D-CNN Backbone
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', name='shared_conv1')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name='shared_pool1')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', name='shared_conv2')(x)
    x = tf.keras.layers.Flatten(name='shared_flatten')(x)
    shared_repr = tf.keras.layers.Dense(64, activation='relu', name='shared_dense')(x)
    
    backbone = tf.keras.Model(inputs=inputs, outputs=shared_repr, name='backbone')
    
    # Task A Head (Regression: Temperature)
    head_a_in = tf.keras.Input(shape=(64,), name='head_a_in')
    a = tf.keras.layers.Dense(32, activation='relu', name='head_a_dense1')(head_a_in)
    out_a = tf.keras.layers.Dense(1, activation=None, name='task_a_out')(a) # Linear for regression
    head_a = tf.keras.Model(inputs=head_a_in, outputs=out_a, name='head_a')
    
    # Task B Head (Classification: Rain)
    head_b_in = tf.keras.Input(shape=(64,), name='head_b_in')
    b = tf.keras.layers.Dense(32, activation='relu', name='head_b_dense1')(head_b_in)
    out_b = tf.keras.layers.Dense(1, activation='sigmoid', name='task_b_out')(b) # Sigmoid for classification
    head_b = tf.keras.Model(inputs=head_b_in, outputs=out_b, name='head_b')
    
    # Full Model
    shared_features = backbone(inputs)
    pred_a = head_a(shared_features)
    pred_b = head_b(shared_features)
    
    full_model = tf.keras.Model(inputs=inputs, outputs=[pred_a, pred_b], name='full_model')
    
    # Attach sub-models for easy access to their trainable variables
    full_model.backbone = backbone
    full_model.head_a = head_a
    full_model.head_b = head_b
    
    return full_model
