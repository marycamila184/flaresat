import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np

LEARNING_RATE = 0.0001
MASK_CHANNELS = 1

def unet_sentinel_forest(input_size, target_size=(512, 512)):
    model_path = '/home/marycamila/flaresat/train/models/transfer_learning/models/unet-sentinel-forest-4c.hdf5'
    old_model = tf.keras.models.load_model(model_path)

    first_layer = old_model.layers[1]
    old_weights = first_layer.get_weights()

    # Create a new Conv2D layer with the correct number of input channels
    new_input = Input(shape=(target_size[0], target_size[1], input_size[2]))

    # Create new Conv2D layer that matches the original layer's output channels
    new_conv_layer = Conv2D(
        filters=old_weights[0].shape[3],
        kernel_size=first_layer.kernel_size,
        strides=first_layer.strides,
        padding=first_layer.padding,
        activation=first_layer.activation
    )(new_input)

    # Adjust the weights to match the new input channels
    new_weights = np.zeros((old_weights[0].shape[0], old_weights[0].shape[1], input_size[2], old_weights[0].shape[3]))
    new_weights[:, :, :old_weights[0].shape[2], :] = old_weights[0]  # Copy weights for existing channels

    # Set the weights for the new Conv2D layer
    new_conv_layer = Conv2D(
        filters=old_weights[0].shape[3],
        kernel_size=first_layer.kernel_size,
        strides=first_layer.strides,
        padding=first_layer.padding,
        activation=first_layer.activation
    )
    new_conv_layer.build(new_input.shape)
    new_conv_layer.set_weights([new_weights, old_weights[1]])

    # Pass the input through the new Conv2D layer
    x = new_conv_layer(new_input)

    old_model.summary()

    for layer in old_model.layers[2:]:
        print(f"\nLayer: {layer.name}")
        try:
            print(f"Expected output shape of layer: {layer.output_shape}")
        except:
            print(f"Layer {layer.name} has dynamic output shape")

        print(f"Input tensor shape before this layer: {x.shape}")

        try:
            x = layer(x)
            print(f"Output tensor shape after this layer: {x.shape}")
        except Exception as e:
            print(f"Error after layer {layer.name}: {e}")
            break  # Stop debugging if there's an error

    new_model = Model(inputs=new_input, outputs=x)

    # Compile the new model
    new_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_focal_crossentropy', metrics=['accuracy'])

    # Print summary of the new model
    print(new_model.summary())

    return new_model

# Call the function with the new input size (512, 512, 10)
unet_sentinel_forest((256, 256, 10), target_size=(512, 512))
