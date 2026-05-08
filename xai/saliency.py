import numpy as np
import tensorflow as tf
import cv2

def get_saliency_map(img_array, model):
    """
    Computes Saliency Map (Pixel Importance) for a given image.
    This is extremely robust as it only requires the input and output nodes.
    """
    img_tensor = tf.convert_to_tensor(img_array)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # Get the gradients of the loss with respect to the input image
    grads = tape.gradient(loss, img_tensor)
    
    # Take the maximum absolute value across color channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    
    # Normalize to [0, 1]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-10)
    
    return saliency.numpy()

def overlay_saliency_map(img_path, saliency, alpha=0.6):
    """
    Overlays the saliency map on the original image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize saliency back to original image size
    saliency_resized = cv2.resize(saliency, (img.shape[1], img.shape[0]))
    
    # Apply color map
    heatmap = np.uint8(255 * saliency_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    
    # Superimpose
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    
    return overlayed
