# Librerías a Importar
import tensorflow as tf
import numpy as np
import cv2
from skimage.io import imread,imshow,imsave
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import keras
 
################# FUNCIONES BASE ######################
# Redimensionar imágenes
def resize_im(im, im_size):
  res_fact = im_size/im.shape[0]
  im_res = rescale(im, res_fact, anti_aliasing = True)
  return im_res
# Padding Cuadrado
def impad(im): 
  # De forma automática hacemos el padding cuadrado sin conocer el tamaño de la imagen original
  s = im.shape
  mx = max(s)
  mx_i = np.argmax(s)

  # Identificamos si alguna dimension es impar
  r0 = s[0] % 2
  r1 = s[1] % 2
  
  # Hacemos a las dimensiones pares si alguna no lo es
  if (r0 == 1):
    im = im[0:s[0]-1,:]
  if (r1 == 1):
    im = im[:,0:s[1]-1]

  # Valor del padding
  pad_val = int((abs(s[mx_i] - s[abs(mx_i-1)]))/2)

  # Condicional para elegir tuplas que indican la direccion del padding, en base al mayor x,y
  if mx_i == 0:
    tup_pad = [(0,0),(pad_val, pad_val)]
  else:
    tup_pad = [(pad_val, pad_val),(0,0)]

  # Hacemos el padding
  imp = np.pad(im, tup_pad)

  return imp, pad_val
# GRADCAM: Último feature Map
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
  # First, we create a model that maps the input image to the activations
  # of the last conv layer as well as the output predictions
  grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

  # Then, we compute the gradient of the top predicted class for our input image
  # with respect to the activations of the last conv layer
  with tf.GradientTape() as tape:
      last_conv_layer_output, preds = grad_model(img_array)
      if pred_index is None:
          pred_index = tf.argmax(preds[0])
      class_channel = preds[:, pred_index]

  # This is the gradient of the output neuron (top predicted or chosen)
  # with regard to the output feature map of the last conv layer
  grads = tape.gradient(class_channel, last_conv_layer_output)

  # This is a vector where each entry is the mean intensity of the gradient
  # over a specific feature map channel
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  # We multiply each channel in the feature map array
  # by "how important this channel is" with regard to the top predicted class
  # then sum all the channels to obtain the heatmap class activation
  last_conv_layer_output = last_conv_layer_output[0]
  heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)

  # For visualization purpose, we will also normalize the heatmap between 0 & 1
  heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
  return heatmap.numpy()
# Realizar GRADCAM y superponer
def create_gradcam(img, heatmap, f_size, val_pad, alpha=0.4):
    # Load the original image
    #img = tf.keras.preprocessing.image.load_img(img_path)
    img = cv2.resize(img*255, (f_size[1], f_size[1]), interpolation = cv2.INTER_AREA) # Resize al tamaño final
    img = Image.fromarray(np.uint8(img))
    img = tf.keras.preprocessing.image.img_to_array(img)


    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img = np.array(superimposed_img.crop((0,val_pad,f_size[1],f_size[1]-val_pad)))

    return superimposed_img


################# FUNCIONES CORE ######################

# PREPROCESAMIENTO
def prepros(im):
    orig_size = im.shape
    imsize = 256
    im,val_pad = impad(im) # Padding para que la imagen sea cuadrada
    im = cv2.resize(im, (imsize, imsize), interpolation = cv2.INTER_AREA) # Resize al tamaño final
    im = im/np.amax(im) # Re-normalizamos
    img = im # Guardamos imagen original
    im = im.reshape((1,) + im.shape) # Agregamos una dimensión para la red
    return im, img, orig_size, val_pad

def Model_predict(model_path, im, img, orig_size, val_pad):
  # Predicción
  model = keras.models.load_model(model_path)
  pred = model.predict(im, verbose = 0)
  pred = np.round(float(pred),decimals = 3)*100

  #Explicación
  model.layers[-1].activation = None #Retiramos la calsificaión(NN)
  # Ya que usamos el feature map final, truncamos el modelo en su última capa
  last_conv_layer_name = 'block14_sepconv2_act' # Nombre depende de cada red, este solo aplica para Xception
  heatmap = make_gradcam_heatmap(im, model, last_conv_layer_name)
  gradcam = create_gradcam(img, heatmap, orig_size, val_pad)

  return pred, gradcam

################# FUNCIÓN MAIN ######################

def DLO_predict(im, mname):
    # Preprocesamiento
    im, img, orig_size, val_pad = prepros(im)
    # Predicción & Explicación
    diag_prob, gradcam = Model_predict(mname, im, img, orig_size, val_pad)

    return diag_prob, gradcam


    

