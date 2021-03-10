import streamlit as st
import adain_style_transfer as ast
from tempfile import NamedTemporaryFile
from PIL import Image
import io

stn = ast.AdainStyleTransfer()

def viewStyleTransfer():
    st.title("Style Transfer")
    alpha = st.slider("Choose alpha value: ", min_value=0.0, max_value=1.0, value=0.5, step=0.1)


    content_file = st.file_uploader("Choose an image for Content", type=["jpg"])
    style_file = st.file_uploader("Choose an image for Style", type=["jpg"])

    col1, col2, col3 = st.beta_columns(3)

    content_image = None
    style_image = None
    
    with col1:
        st.header("Content")
        if(content_file is not None):
            image = content_file.read()
            img = st.image(image, caption='your content', use_column_width=True)
            #st.write(content_file.name)
            content_image=image

                #temp_file = NamedTemporaryFile(delete=False)
                #temp_file.write(image)
                #print(temp_file.name)
    with col2:
        st.header("Style")    
        if(style_file is not None):
            image = style_file.read()
            img = st.image(image, caption='your style', use_column_width=True)
            st.write(style_file.name)
            style_image=image
    with col3:
        st.header("Result")
        if(content_file is not None and style_file is not None):
            output = stn.style_transfer( Image.open(io.BytesIO(content_image)),  Image.open(io.BytesIO(style_image)), alpha)
            image=Image.fromarray(output)
            st.image(image, caption='transferred image', use_column_width=True)
                        

def main():
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "StyleTransfer", "Segmentation"])

    if(page == "Homepage"):
        st.header("About")
        st.write("Heal the world")
        viewStyleTransfer()
    elif(page == "StyleTransfer"):
        viewStyleTransfer()
    elif(page == "Segmentation"):        
        st.title("TBD")

    stn.style_transfer_file("./data/content_img_1.jpg", "./data/style_img_1.jpg", 0.5)        

if __name__ == "__main__":
    main()
