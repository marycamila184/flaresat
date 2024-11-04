import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.001
MASK_CHANNELS = 1

def decoder_block(x, skip_connection, num_filters):
    x = layers.Conv2DTranspose(num_filters, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.concatenate([x, skip_connection])
    x = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    return x

def unet_with_resnet(input_size):
    inputs = layers.Input(shape=input_size)

    if input_size[-1] != 3:
        x = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(inputs)
    else:
        x = inputs

    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=x)

    skip_connection_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
    skip_connections = [base_model.get_layer(name).output for name in skip_connection_names]

    # Define the ResNet output (encoder output)
    resnet_output = base_model.output

    # Decoder path using decoder_block function
    x = decoder_block(resnet_output, skip_connections[-1], 1024)
    x = decoder_block(x, skip_connections[-2], 512)
    x = decoder_block(x, skip_connections[-3], 256)
    x = decoder_block(x, skip_connections[-4], 64)

    # Final layer with the number of mask channels
    x = layers.Conv2DTranspose(64, 2, activation='relu', strides=2, padding='same')(x)
    outputs = layers.Conv2D(MASK_CHANNELS, 1, activation='sigmoid')(x)

    # Build the final model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_focal_crossentropy', metrics=['accuracy'])

    return model