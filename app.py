import streamlit as st

page_bg = """
<style>
[data-testid = "stAppViewContainer"]{
background-image: url("https://img.freepik.com/free-photo/abstract-orange-paint-background-acrylic-texture-with-marble-pattern_1258-90489.jpg?w=2000");
background-size: cover;
}
</style>
"""


st.markdown(page_bg, unsafe_allow_html = True)
st.title('Welcome to DeepLearningOp')



# def load_image():
#     uploaded_file = st.file_uploader(label='Pick an image to test')
#     if uploaded_file is not None:
#         image_data = uploaded_file.getvalue()
#         st.image(image_data)        


# def load_labels():
#     labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
#     labels_file = os.path.basename(labels_path)
#     if not os.path.exists(labels_file):
#         wget.download(labels_path)
#     with open(labels_file, "r") as f:
#         categories = [s.strip() for s in f.readlines()]
#         return categories
        

# def main():
#     st.title('DeepLungOp - DEMO')
#     st.header('DLO Computer Aided Diagnosis')
#     st.write('Somos tu confiable asistente, la segunda opinión que buscas ..')
#     load_image()


# if __name__ == '__main__':
#     main()

