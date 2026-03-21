import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# 1. LOAD MODEL
model = VGG16(weights='imagenet')


# 2. DEFINE GRAD-CAM LOGIC
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam_overlay(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = (jet_heatmap * alpha) + img
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Activation Map")
    plt.axis('off')
    plt.show()


# 3. DEFINE PREDICTION FUNCTION
def predict_and_visualize(img_path):
    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)
    results = decode_predictions(preds, top=3)[0]

    print(f"--- Top 3 Predictions for {img_path} ---")
    for _, label, score in results:
        print(f"{label}: {score * 100:.2f}%")

    # Generate and Show Heatmap
    last_layer = "block5_conv3"
    heatmap = make_gradcam_heatmap(x, model, last_layer)
    display_gradcam_overlay(img_path, heatmap)


# 4. EXECUTION
# This is the ONLY place where your specific file path should go.
my_rabbit_path = r'C:\Users\gavin\Pictures\rabbit.png'
predict_and_visualize(my_rabbit_path)