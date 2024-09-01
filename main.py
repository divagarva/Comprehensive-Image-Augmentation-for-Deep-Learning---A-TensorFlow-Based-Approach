import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

# Relative path to the dataset
PATH = os.path.join(os.getcwd(), "cat_dog_sub/test")

# Load the dataset
ds = image_dataset_from_directory(
    PATH,
    validation_split=0.2,
    subset="training",
    image_size=(256, 256),
    interpolation="bilinear",
    crop_to_aspect_ratio=True,
    seed=42,
    shuffle=True,
    batch_size=10
)

# Take one batch from the dataset and display the images
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5, 5))
for images, labels in ds.take(1):
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(images[i * 3 + j].numpy().astype("uint8"))
            ax[i][j].set_title(ds.class_names[labels[i * 3 + j]])
            ax[i][j].axis('on')  # Show axes
plt.show()

# Create preprocessing layers
out_height, out_width = 128, 256
resize = tf.keras.layers.Resizing(out_height, out_width)
height = tf.keras.layers.RandomHeight(0.3)
width = tf.keras.layers.RandomWidth(0.3)
zoom = tf.keras.layers.RandomZoom(0.3)

flip = tf.keras.layers.RandomFlip("horizontal_and_vertical")
rotate = tf.keras.layers.RandomRotation(0.2)
crop = tf.keras.layers.RandomCrop(out_height, out_width)
translation = tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)

brightness = tf.keras.layers.RandomBrightness([-0.8, 0.8])
contrast = tf.keras.layers.RandomContrast(0.2)

# Visualize Resizing, Height, Width, and Zoom augmentations
fig, ax = plt.subplots(5, 3, figsize=(6, 14))
for images, labels in ds.take(1):
    for i in range(3):
        # Convert images to float32
        img_float32 = tf.image.convert_image_dtype(images[i], dtype=tf.float32)

        ax[0][i].imshow(images[i].numpy().astype("uint8"))
        ax[0][i].set_title("original")

        # Resize
        resized_img = resize(img_float32)
        ax[1][i].imshow(tf.clip_by_value(resized_img * 255, 0, 255).numpy().astype("uint8"))
        ax[1][i].set_title("resize")

        # Height
        height_img = height(img_float32)
        ax[2][i].imshow(tf.clip_by_value(height_img * 255, 0, 255).numpy().astype("uint8"))
        ax[2][i].set_title("height")

        # Width
        width_img = width(img_float32)
        ax[3][i].imshow(tf.clip_by_value(width_img * 255, 0, 255).numpy().astype("uint8"))
        ax[3][i].set_title("width")

        # Zoom
        zoom_img = zoom(img_float32)
        ax[4][i].imshow(tf.clip_by_value(zoom_img * 255, 0, 255).numpy().astype("uint8"))
        ax[4][i].set_title("zoom")
plt.show()

# Visualize Flip, Crop, Translation, and Rotate augmentations
fig, ax = plt.subplots(5, 3, figsize=(6, 14))
for images, labels in ds.take(1):
    for i in range(3):
        ax[0][i].imshow(images[i].numpy().astype("uint8"))
        ax[0][i].set_title("original")

        # Flip
        ax[1][i].imshow(flip(images[i]).numpy().astype("uint8"))
        ax[1][i].set_title("flip")

        # Crop
        ax[2][i].imshow(crop(images[i]).numpy().astype("uint8"))
        ax[2][i].set_title("crop")

        # Translation
        ax[3][i].imshow(translation(images[i]).numpy().astype("uint8"))
        ax[3][i].set_title("translation")

        # Rotate
        ax[4][i].imshow(rotate(images[i]).numpy().astype("uint8"))
        ax[4][i].set_title("rotate")
plt.show()

# Visualize Brightness and Contrast augmentations
fig, ax = plt.subplots(3, 3, figsize=(6, 8))
for images, labels in ds.take(1):
    for i in range(3):
        ax[0][i].imshow(images[i].numpy().astype("uint8"))
        ax[0][i].set_title("original")

        # Brightness
        ax[1][i].imshow(brightness(images[i]).numpy().astype("uint8"))
        ax[1][i].set_title("brightness")

        # Contrast
        ax[2][i].imshow(contrast(images[i]).numpy().astype("uint8"))
        ax[2][i].set_title("contrast")
plt.show()
