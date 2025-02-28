import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image, ImageSequence

# Set Page Title and Layout
st.set_page_config(page_title="Image Filter & GIF Generator", page_icon="🎨", layout="centered")

# Title
st.title("🎨 Image Filter & GIF Generator")
st.write("Upload an image, apply artistic filters, and create animated GIFs!")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

# Filter selection
filter_option = st.selectbox("🖌️ Choose a filter:", 
                             ["Pencil Sketch", "Doodling", "Watercolor", "Cartoon", "Thermal Vision",
                              "Oil Painting", "Pencil Color Sketch", "Emboss", "Sepia", "HDR"])

# Function: Apply Filters
def apply_filter(image, filter_type):
    if image is None:
        return None

    if filter_type == "Pencil Sketch":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blurred = cv2.bitwise_not(blurred)
        return cv2.divide(gray, inverted_blurred, scale=256.0)
    
    elif filter_type == "Doodling":
        return cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    
    elif filter_type == "Watercolor":
        return cv2.bilateralFilter(image, 15, 75, 75)
    
    elif filter_type == "Cartoon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 250, 250)
        return cv2.bitwise_and(color, color, mask=edges)

    elif filter_type == "Thermal Vision":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    elif filter_type == "Oil Painting":
        return cv2.bilateralFilter(image, 9, 75, 75)

    elif filter_type == "Pencil Color Sketch":
        gray, color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return color
    
    elif filter_type == "Emboss":
        kernel = np.array([[0, -1, -1],
                           [1,  0, -1],
                           [1,  1,  0]])
        return cv2.filter2D(image, -1, kernel)
    
    elif filter_type == "Sepia":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia = cv2.transform(image, sepia_filter)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    
    elif filter_type == "HDR":
        return cv2.detailEnhance(image, sigma_s=12, sigma_r=0.15)

    return image

# Function: Convert Filtered Image to Animated GIF
def create_gif(image):
    frames = [image.point(lambda p: p * (1 + i * 0.02)) for i in range(10)]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    frames[0].save(temp_file.name, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return temp_file.name

# Process image when uploaded
if uploaded_file is not None:
    # Convert uploaded image to PIL format
    image = Image.open(uploaded_file)

    # Convert image to NumPy array (RGB)
    image = np.array(image)

    # Convert RGB to BGR for OpenCV processing
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Show original image (convert back to RGB for correct display)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="📷 Original Image", use_container_width=True)

    # Apply selected filter
    if st.button("🎨 Generate Image"):
        output = apply_filter(image, filter_option)

        if output is not None:
            # Convert grayscale images to RGB
            if len(output.shape) == 2:
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

            # Convert to PIL format for GIF processing
            pil_output = Image.fromarray(output)

            # Create a GIF
            gif_path = create_gif(pil_output)

            # Display Filtered Image & GIF Side by Side
            col1, col2 = st.columns(2)

            with col1:
                st.image(output, caption=f"🖼️ {filter_option} Effect", use_container_width=True)
                
                # Save image to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(temp_file.name, output)

                # Centered Download Button for Image
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.download_button("💾 Download Image", data=open(temp_file.name, "rb").read(),
                                   file_name=f"{filter_option.lower().replace(' ', '_')}.jpg", mime="image/jpeg")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.image(gif_path, caption="🎞️ Animated GIF", use_container_width=True)

                # Centered Download Button for GIF
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.download_button("🎞️ Download GIF", data=open(gif_path, "rb").read(),
                                   file_name=f"{filter_option.lower().replace(' ', '_')}.gif", mime="image/gif")
                st.markdown("</div>", unsafe_allow_html=True)
