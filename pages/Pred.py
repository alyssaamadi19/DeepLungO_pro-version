import streamlit as st
from tempfile import NamedTemporaryFile
import numpy as np
from io import StringIO

import sys
sys.path.insert(1, 'C:/Users/alyssa/OneDrive/Escritorio/Streamlit/Streamlit') 
import CAD_DLOv1 as cad

page_bg = """
<style>
[data-testid = "stAppViewContainer"]{
background-image: url("https://img.freepik.com/free-photo/abstract-orange-paint-background-acrylic-texture-with-marble-pattern_1258-90489.jpg?w=2000");
background-size: cover;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html = True)

st.markdown("""
    # Predicción de Radiografía
    Nuestra IA utiliza una Red Neuronal Convolucional del tipo Xception, entrenada miles de radiografías provenientes de bases de datos de grado médico.

    ### A continuación sube la radiografía:
""")

upload_file = st.file_uploader('Arrastra o selecciona el archivo de tu computadora')

if upload_file is not None:
    #st.image(upload_file)
    x= upload_file.read()
    st.write(stringio)
    #st.image(im)

    # model_path = 'Xception.h5'
    # pred, gradcam = cad.DLO_predict(upload_file, model_path)
    
    # st.markdown("""
    #     # Probabilidad de Opacidad Pulmonar:{} %
    # """.format(pred))

    # st.image(gradcam)
