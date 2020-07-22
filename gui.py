import streamlit as st
from PIL import Image
from LSTM import LSTM

lstm = LSTM()

chosen_image = None

img1 = Image.open('images/sunrise.png')
img2 = Image.open('images/men-football.png')
img3 = Image.open('images/man-dirtbike.png')
img4 = Image.open('images/two-dogs.png')
img5 = Image.open('images/white-bird.png')

st.title('Image Captioning')
st.markdown('''
    This is a Deep Learning Model that will try to Caption any Image. Check out the Notebook used to train this 
    [here](https://github.com/saksham2001).
''')

chosen_model = st.selectbox('Choose the Language Model to be used?', ('LSTM', 'Bert', 'GPT-3'))

st.write('You selected: ', chosen_model)

img_type = st.radio('Image', ('See Image Examples from the Dataset', 'Upload your own Image'))

if chosen_model == 'LSTM':
    if img_type == 'See Image Examples from the Dataset':
        st.image(img1, caption='Sunrise by the mountains', use_column_width=True)
        st.image(img2, caption='Two Men are playing Football in the field.', use_column_width=True)
        st.image(img3, caption='Man on Bike is riding on the Dirt Bike.', use_column_width=True)
        st.image(img4, caption='Two Dogs are running in the Dry.', use_column_width=True)
        st.image(img5, caption='A White Bird in Water.', use_column_width=True)
    elif img_type == 'Upload your own Image':
        chosen_image = st.file_uploader(label='Upload any jpg image here!', type='jpg')

    if chosen_image is not None:
        st.image(chosen_image, caption='Uploaded Image', use_column_width=True)
        with st.spinner('Processing Image...'):
            caption = lstm.predict_caption(chosen_image)
            st.markdown(f'''
                ## {caption}
                ''')
            st.balloons()
elif chosen_model == 'Bert':
    st.markdown('''
    # Bert is currently Not Supported
    ''')
elif chosen_model == 'GPT-3':
    st.markdown('''
        # GPT-3 is currently Not Supported
        ''')

st.markdown('#### Developed by [Saksham Bhutani](https://github.com/saksham2001).')
