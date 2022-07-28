import tensorflow as tf

def net(base_model, data_augmentation, img_height, img_width):
    # Create new model on top
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet.preprocess_input(x, data_format=None)   # will convert the input images from RGB to BGR, 
                                                                                    #then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(x, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x) # Regularize with dropout
    # A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model