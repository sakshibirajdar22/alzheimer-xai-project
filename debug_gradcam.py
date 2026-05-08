import tensorflow as tf
import numpy as np
import os
from xai.gradcam import get_gradcam_heatmap

def debug_gradcam():
    if not os.path.exists('model/model.h5'):
        print("Model not found.")
        return
        
    model = tf.keras.models.load_model('model/model.h5')
    dummy_img = np.random.random((1, 224, 224, 3)).astype('float32')
    
    # Try custom CNN layer name or ResNet name
    is_resnet = any('resnet50' in layer.name.lower() for layer in model.layers)
    if is_resnet:
        layer_name = 'conv5_block3_out'
    else:
        layer_name = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1]
        
    print(f"Testing Grad-CAM with layer: {layer_name}")
    try:
        heatmap = get_gradcam_heatmap(dummy_img, model, layer_name)
        print("Success! Heatmap shape:", heatmap.shape)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gradcam()
