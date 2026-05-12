import tensorflow as tf

def build_mtl_model(input_shape=(7, 3)):
    """Builds an LSTM multi-task model for spatio-temporal data."""
    inputs = tf.keras.Input(shape=input_shape, name='input_layer')
    
    # Shared LSTM Backbone
    x = tf.keras.layers.LSTM(64, return_sequences=True, name='shared_lstm1')(inputs)
    x = tf.keras.layers.LSTM(32, name='shared_lstm2')(x)
    shared_repr = tf.keras.layers.Dense(64, activation='relu', name='shared_dense')(x)
    
    backbone = tf.keras.Model(inputs=inputs, outputs=shared_repr, name='backbone')
    
    # Task A Head (Regression: Temperature)
    head_a_in = tf.keras.Input(shape=(64,), name='head_a_in')
    a = tf.keras.layers.Dense(32, activation='relu', name='head_a_dense1')(head_a_in)
    a = tf.keras.layers.BatchNormalization(name='head_a_bn')(a)
    a = tf.keras.layers.Dropout(0.2, name='head_a_dropout')(a)
    out_a = tf.keras.layers.Dense(1, activation=None, name='task_a_out')(a) # Linear for regression
    head_a = tf.keras.Model(inputs=head_a_in, outputs=out_a, name='head_a')
    
    # Task B Head (Classification: Rain)
    head_b_in = tf.keras.Input(shape=(64,), name='head_b_in')
    b = tf.keras.layers.Dense(32, activation='relu', name='head_b_dense1')(head_b_in)
    b = tf.keras.layers.BatchNormalization(name='head_b_bn')(b)
    b = tf.keras.layers.Dropout(0.2, name='head_b_dropout')(b)
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
