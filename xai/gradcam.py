import numpy as np
import tensorflow as tf
import cv2

def get_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Reconstructs the model as a Functional graph to ensure all layers are 
    properly 'called' and linked for GradientTape. Handles nested layers.
    """
    
    # 1. Reconstruct a Functional model from layers to ensure connectivity
    # This is the most robust way to fix 'layer never been called' errors.
    inputs = tf.keras.Input(shape=model.input_shape[1:])
    x = inputs
    
    layer_outputs = {}
    
    # Iterate through layers and build the graph
    for layer in model.layers:
        # Check if this layer is a container (like ResNet50 base)
        if hasattr(layer, 'layers'):
            # It's a sub-model, we need to go inside to find the target layer
            # But we call the whole sub-model first to keep it simple
            sub_x = layer(x)
            
            # If the target layer is inside this sub-model, we need a separate path
            # to capture that specific intermediate output.
            if any(l.name == last_conv_layer_name for l in layer.layers):
                # We create a internal functional model for the sub-model
                sub_model = layer
                sub_inputs = tf.keras.Input(shape=sub_model.input_shape[1:])
                s_x = sub_inputs
                target_out = None
                for sl in sub_model.layers:
                    s_x = sl(s_x)
                    if sl.name == last_conv_layer_name:
                        target_out = s_x
                
                # Intermediate model for the submodel
                internal_m = tf.keras.Model(sub_inputs, [target_out, s_x])
                conv_out, x = internal_m(x)
                layer_outputs[last_conv_layer_name] = conv_out
            else:
                x = sub_x
        else:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                layer_outputs[last_conv_layer_name] = x

    # Final reconstructed model that outputs what we need
    if last_conv_layer_name not in layer_outputs:
         raise ValueError(f"Layer {last_conv_layer_name} not found.")
         
    grad_model = tf.keras.Model(inputs, [layer_outputs[last_conv_layer_name], x])

    # 2. Gradient Calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Gradient of the class output w.r.t. the conv layer feature maps
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Standard visualization.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    return superimposed_img
