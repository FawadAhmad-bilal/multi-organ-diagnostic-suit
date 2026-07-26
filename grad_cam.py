import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
   
    grad_model = keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    
    grads = tape.gradient(class_channel, conv_output)

    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

   
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

  
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index


def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap="jet"):
    
    heatmap_uint8 = np.uint8(255 * heatmap)

    
    cmap = matplotlib.colormaps[colormap]
    cmap_colors = cmap(np.arange(256))[:, :3]
    colored_heatmap = cmap_colors[heatmap_uint8]

    colored_heatmap = keras.utils.array_to_img(colored_heatmap)
    colored_heatmap = colored_heatmap.resize(
        (original_img.shape[1], original_img.shape[0])
    )
    colored_heatmap = keras.utils.img_to_array(colored_heatmap)

    superimposed_img = colored_heatmap * alpha + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    return superimposed_img
