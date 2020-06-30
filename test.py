from flask import Flask,render_template,request,send_file,jsonify
from flask_cors import CORS
import functools
import os
from matplotlib import gridspec
import matplotlib.pylab as plt
import matplotlib.pyplot as plt1
import numpy as np
import tensorflow as tf
import io
from PIL import Image
config =  tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True
sess =  tf.compat.v1.Session(config=config)
import tensorflow_hub as hub
from PIL import Image
import base64
from matplotlib import cm
print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

# @title Define image loading and visualization functions  { display-mode: "form" }


app = Flask(__name__)
CORS(app)

output_image_size = 256  # @param {type:"integer"}
def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w  * n, w))
  # gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    # plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()

"""Let's get as well some images to play with."""

# @title Load example images  { display-mode: "form" }

# content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg'  # @param {type:"string"}
# style_image_url = 'https://www.byhien.com/wp-content/uploads/2020/04/art-painting-abstract.jpg'  # @param {type:"string"}
# output_image_size = 256  # @param {type:"integer"}

# image_path1 = tf.keras.utils.get_file(os.path.basename(style_image_url)[-128:], style_image_url)
# image_path2='D:\WhatsApp Image 2020-06-30 at 3.11.05 PM.jpeg'
# The content image size can be arbitrary.
# content_img_size = (output_image_size, output_image_size)
# The style prediction model was trained with image size 256 and it's the 
# recommended image size for the style image (though, other sizes work as 
# well but will lead to different results).
# style_img_size = (256, 256)  # Recommended to keep it at 256.

# content_image = load_image(image_path2, content_img_size)
# style_image = load_image(image_path1, style_img_size)
# style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
# show_n([content_image, style_image], ['Content image', 'Style image'])

"""## Import TF-Hub module"""

# Load TF-Hub module.

# hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
# hub_module = hub.load(hub_handle)

"""The signature of this hub module for image stylization is:
```
outputs = hub_module(content_image, style_image)
stylized_image = outputs[0]
```
Where `content_image`, `style_image`, and `stylized_image` are expected to be 4-D Tensors with shapes `[batch_size, image_height, image_width, 3]`.

In the current example we provide only single images and therefore the batch dimension is 1, but one can use the same module to process more images at the same time.

The input and output values of the images should be in the range [0, 1].

The shapes of content and style image don't have to match. Output image shape
is the same as the content image shape.

## Demonstrate image stylization
"""

# Stylize content image with given style image.
# This is pretty fast within a few milliseconds on a GPU.

# outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
# stylized_image = outputs[0]

# Visualize input images and the generated stylized image.

# show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])




@app.route("/")
def root():
    return "Server at http://localhost:5000"


@app.route('/pic',methods=['GET'])

def the_main_thing():
      field=field=request.args
      content_image_url = field['content']  # @param {type:"string"}
      style_image_url = field['style']  # @param {type:"string"}
      output_image_size = 256  # @param {type:"integer"}
      image_path1 = tf.keras.utils.get_file(os.path.basename(style_image_url)[-128:], style_image_url)
      image_path2 = tf.keras.utils.get_file(os.path.basename(content_image_url)[-128:], content_image_url)
      content_image=load_image(image_path2)
      style_image=load_image(image_path1)
      hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
      hub_module = hub.load(hub_handle)
      outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
      stylized_image = outputs[0]
      # im = Image.fromarray(np.array(stylized_image[0])*255)
      # return send_file(im)
      stylized_image=np.array(stylized_image[0])*255
      img = Image.fromarray(stylized_image.astype("uint8"))
      newsize = (400, 400) 
      img = img.resize(newsize) 
      print(field['content'],field['style'])
      rawBytes = io.BytesIO()
      img.save(rawBytes, "JPEG")
      rawBytes.seek(0)
      return send_file(rawBytes,attachment_filename='st_act.png',mimetype='image/png')



