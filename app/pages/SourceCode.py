import streamlit as st
from PIL import Image
from streamlit_extras.app_logo import add_logo

st.set_page_config(
    page_title="Eyedentify", 
    page_icon=":eye:"
    )  # Set icon to None to hide default

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# add_logo("app/logo4.png")

with st.sidebar:
    st.image("app/logo4.png", use_column_width=True)
    st.caption("By Salman Gassem © 2024")

st.title("Source Code")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dependencies", "Training", "Evaluation", "Testing", "App Layout"])

with tab1:

    st.write("The language used is Python.")

    libraries = [
    ("TensorFlow, version 2.15.0", "[TensorFlow](https://www.tensorflow.org/)", "A powerful open-source library for numerical computation and large-scale machine learning."),
    ("Google Colab", "[Google Colab](https://colab.google/)", "A free Jupyter notebook environment that runs in the cloud."),
    ("PIL (Python Imaging Library)", "[PIL](https://pillow.readthedocs.io/)", "A popular library for working with images in Python."),
    ("Matplotlib", "[Matplotlib](https://matplotlib.org/)", "A comprehensive library for creating static, animated, and interactive visualizations in Python."),
    ("NumPy (Numerical Python)", "[NumPy](https://numpy.org/)", "A fundamental library for scientific computing in Python."),
    ("Streamlit", "[Streamlit](https://docs.streamlit.io/)", "A user-friendly library for creating web applications in Python."),
    ("cv2 (OpenCV)", "[OpenCV](https://opencv.org/)", "A powerful library for computer vision tasks like image processing, object detection, and video analysis."),
    ("time", "[Built-in Python module](https://docs.python.org/3/library/time.html)", "Provides functionalities for working with time and dates."),
    ("os", "[Built-in Python module](https://docs.python.org/3/library/os.html)", "Provides functionalities for interacting with the operating system."),
    ("imghdr", "[Built-in Python module](https://docs.python.org/3/library/imghdr.html)", "Helps determine the image format of a file based on its header information."),
]

    for name, link, description in libraries:
        st.subheader(name)
        st.markdown(f"{link} - {description}")
        st.write("---")

with tab2:
    st.subheader("Imports")
    code = '''import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Dataset Preparation")
    code = '''# Directory of the dataset
data_dir = 'DIRECTORY_PATH'

# creating tensorflow dataset from dataset directory
data = tf.keras.utils.image_dataset_from_directory(data_dir)

# creating dataset iterator to iterate through the data
data_iterator = data.as_numpy_iterator()

# to output the images to check labeling
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
  ax[idx].imshow(img.astype(int))
  ax[idx].title.set_text(batch[1][idx])'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
    
    st.subheader("Preprocessing")
    code = '''# scaling the pixel values of the images from 0.0 to 1.0
data = data.map(lambda x,y: (x/255, y))

# splitting the data into training, validation, and testing datasets
# numbers of batches is 58, len(data)
train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)+1
test_size = int(len(data)*0.1)+1

# assigning dataset batches and preloading them to cache
train = data.take(train_size)
train = train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

val = data.skip(train_size).take(val_size)
val = val.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

test = data.skip(train_size+val_size).take(test_size)
test = test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Model")
    code = '''from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import layers

# creating data augmentation
data_augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
])

# assigning l2 regularizer value
reg_factor = 0.01

# initializing the model
model = Sequential()

# augmentation layer
model.add(data_augmentation)

# convolutional layer 1
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3), kernel_regularizer=l2(reg_factor)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# convolutional layer 2
model.add(Conv2D(32, (3,3), 1, activation='relu', kernel_regularizer=l2(reg_factor)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# convolutional layer 3
model.add(Conv2D(64, (3,3), 1, activation='relu', kernel_regularizer=l2(reg_factor)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# convolutional layer 4
model.add(Conv2D(16, (3,3), 1, activation='relu', kernel_regularizer=l2(reg_factor)))
model.add(BatchNormalization())
model.add(MaxPooling2D())

# Dropout layer, 50%
model.add(Dropout(0.5))

# Flatten layer
model.add(Flatten())

model.add(BatchNormalization())

# Dense layer 1
model.add(Dense(256, activation='relu', kernel_regularizer=l2(reg_factor)))

# Dense layer 2
model.add(Dense(1, activation='sigmoid'))

# compiling the model
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
    
    st.subheader("Model Summary")
    code = '''Model: "sequential_66"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 sequential_65 (Sequential)  (None, 256, 256, 3)       0         
                                                                 
 conv2d_189 (Conv2D)         (None, 254, 254, 16)      448       
                                                                 
 batch_normalization_57 (Ba  (None, 254, 254, 16)      64        
 tchNormalization)                                               
                                                                 
 max_pooling2d_188 (MaxPool  (None, 127, 127, 16)      0         
 ing2D)                                                          
                                                                 
 conv2d_190 (Conv2D)         (None, 125, 125, 32)      4640      
                                                                 
 batch_normalization_58 (Ba  (None, 125, 125, 32)      128       
 tchNormalization)                                               
                                                                 
 max_pooling2d_189 (MaxPool  (None, 62, 62, 32)        0         
 ing2D)                                                          
                                                                 
 conv2d_191 (Conv2D)         (None, 60, 60, 64)        18496     
                                                                 
 batch_normalization_59 (Ba  (None, 60, 60, 64)        256       
 tchNormalization)                                               
                                                                 
 max_pooling2d_190 (MaxPool  (None, 30, 30, 64)        0         
 ing2D)                                                          
                                                                 
 conv2d_192 (Conv2D)         (None, 28, 28, 16)        9232      
                                                                 
 batch_normalization_60 (Ba  (None, 28, 28, 16)        64        
 tchNormalization)                                               
                                                                 
 max_pooling2d_191 (MaxPool  (None, 14, 14, 16)        0         
 ing2D)                                                          
                                                                 
 dropout_30 (Dropout)        (None, 14, 14, 16)        0         
                                                                 
 flatten_54 (Flatten)        (None, 3136)              0         
                                                                 
 batch_normalization_61 (Ba  (None, 3136)              12544     
 tchNormalization)                                               
                                                                 
 dense_108 (Dense)           (None, 256)               803072    
                                                                 
 dense_109 (Dense)           (None, 1)                 257       
                                                                 
=================================================================
Total params: 849201 (3.24 MB)
Trainable params: 842673 (3.21 MB)
Non-trainable params: 6528 (25.50 KB)
_________________________________________________________________'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Training step")
    code = '''# initializing a Learning rate scheduler
# dynamically lowers learning rate when no improvement is detected
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)

# to start training the model and saving the results to the variable hist
hist = model.fit(
    train,
    epochs=200,
    validation_data=val,
    callbacks=[reduce_lr]
    )'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Plotting the Performance")
    code = '''acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(200)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
        image = Image.open("app/perplot.png")
        st.image(image, use_column_width=True)

with tab3:

    st.subheader("Metrics used:")
    st.markdown("##### *Precision*")
    st.write("High precision indicates a low rate of false positives.")
    st.markdown("##### *Recall*")
    st.write("High recall indicates that the classes are correctly recognized.")
    st.markdown("##### *BinaryAccuracy*")
    st.write("Measures the percentage of correct predictions out of all predictions made.")

    st.divider()

    st.subheader("Evaluating the Performance")
    code = '''from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# initializing variable with the methods/metrics used to evaluate
precision_eval = Precision()
recall_eval = Recall()
accuracy_eval = BinaryAccuracy()

# running the testing on the unseen test dataset
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision_eval.update_state(y, yhat)
    recall_eval.update_state(y, yhat)
    accuracy_eval.update_state(y, yhat)

# printing and formatting the result
print(f'Precision: {precision_eval.result().numpy():.2f}, Recall: {recall_eval.result().numpy():.2f}, Accuracy: {accuracy_eval.result().numpy():.2f}')'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
    
    st.subheader("Output")
    code = '''Precision: 0.97, Recall: 0.89, Accuracy: 0.94
'''
    with st.expander("Expand Code"):
        st.code(code, language='python')

with tab4:
    st.subheader("Testing the model")
    code = '''import cv2

# assigning an image to a variable
img = cv2.imread('PATH_TO_IMAGE')

# this is just to print the image to check if its loading correctly
# not part of model testing
# also to convert it to RGB from BGR because opencv reads it as BGR
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# preprocessing to the correct model input size
resize = tf.image.resize(img, (256, 256))

# running the prediction test while also scaling it to values of 0.0 to 1.0
# and ensuring the batch dimensions are correct
yhat = model.predict(np.expand_dims(resize/255, 0))

# since our classes values are 0 for defective and 1 for good,
# prediction is < 0.5 = defective cause rounded down to 0,
# prediction is > 0.5 = good cause rounded up to 1
if yhat < 0.5:
    print('Tyre condition: Defective')
else:
    print('Tyre condition: Good')'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

with tab5:
    st.subheader("Home Page")
    code = '''import streamlit as st
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
import time
from streamlit_extras.app_logo import add_logo

st.set_page_config(
    page_title="Eyedentify", 
    page_icon=":eye:"
    ) 

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar:
      st.container(height=900, border=False)
      st.caption("By Salman Gassem © 2024")

add_logo("Logo4.png")

st.title(":eye: Eyedentify")

st.markdown("### Is your tyre ok?\n #### Find out by uploading an image of your tyre! :point_down:")

st.divider()

supported_types = ["jpg", "jpeg", "png"]

uploaded_file = st.file_uploader("Upload an image:")

if uploaded_file is not None:
    file_type = uploaded_file.type.lower().split("/")[-1]
    if file_type not in supported_types:
        with st.status("Receiving image...", expanded=True) as status:
                st.write("Checking image type...")
                time.sleep(0.5)
        st.error(f"Unsupported file type '{file_type}'. Please upload JPG, JPEG, or PNG.")
    else:
        if uploaded_file:
            with st.status("Receiving image...", expanded=True) as status:
                st.write("Checking image type...")
                time.sleep(1)
                st.write("Confirmed!")
                time.sleep(1)
                st.write("Uploading...")
                time.sleep(1)
                status.update(label="Image received!", state="complete", expanded=False)

            time.sleep(1)

            # Model
            if uploaded_file is not None:

                st.header('Prediction:')

                with st.status("Sending image to model...", expanded=True) as status:
                    st.write("Preprocessing...")
                    time.sleep(1)
                    st.write("Inserting to model...")
                    time.sleep(1)
                    st.write("Predicting...")
                    time.sleep(1)
                    status.update(label="Prediction complete!", state="complete", expanded=False)

                if 'loaded_model' not in st.session_state:  # Check if model is already loaded
                    loaded_model = tf.keras.models.load_model("savedmodel")
                    st.session_state['loaded_model'] = loaded_model
                else:
                    loaded_model = st.session_state['loaded_model']
                
                image = Image.open(uploaded_file)

                if image is not None:
                    image = image.convert("RGB")
                    resize = tf.image.resize(image, (256, 256))

                yhat = loaded_model.predict(np.expand_dims(resize/255, 0))

                st.divider()

                with st.spinner('Printing result...'):
                    time.sleep(2)

                if yhat < 0.5:
                    st.error('Tyre condition: Defective')
                else:
                    st.success('Tyre condition: Good')

                st.image(image, caption="Uploaded Image", use_column_width=True)

'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("About Page")
    code = '''import streamlit as st
from streamlit_extras.app_logo import add_logo

st.set_page_config(
    page_title="Eyedentify", 
    page_icon=":eye:"
    )

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

add_logo("Logo4.png")

with st.sidebar:
      st.container(height=900, border=False)
      st.caption("By Salman Gassem © 2024")

with st.container():
    
    
    st.image("log_mid_border.png", use_column_width=True)

    st.header("About Eyedentify")
    st.write("Eyedentify is a project aimed at utilizing image classification techniques to address practical and meaningful challenges in various domains starting with industrial components quality control.")
    
    st.subheader("Why Eyedentify is the future")
    st.write("- Potential to revolutionize the way we solve problems.\n- Recognize and categorize objects, patterns, or features in images.\n- Automated disease diagnosis, object recognition in autonomous vehicles, and even facial recognition for security systems.\n- Machine learning algorithms and approaches that demonstrate accuracy and efficiency.")

    st.divider()

    st.subheader("Objective")
    st.write("Develop an Image Classification system capable of accurately identifying and categorizing manufacturing defects in factory-produced items.")

    st.subheader("To achieve")
    st.write("- Increased manufacturing profits and production quality.\n- Improve efficiency and reduce costs.\n- Enhance the manufacturer’s competitive advantage in the market.\n- Eliminate Human Error.\n- Reduce risk.")

'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Source Code Page")
    st.write("*Due to issues in syntax, placeholders were put in every 'code' variable. These variables can be found from other code snippets of other pages if required, however, code structure remains the same.*")
    code = '''import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Eyedentify", 
    page_icon=":eye:"
    )  # Set icon to None to hide default

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

add_logo("Logo4.png")

with st.sidebar:
      st.container(height=900, border=False)
      st.caption("By Salman Gassem © 2024")

st.title("Source Code")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dependencies", "Training", "Evaluation", "Testing", "App Layout"])

with tab1:

    st.write("The language used is Python.")

    libraries = [
    ("TensorFlow, version 2.15.0", "[TensorFlow](https://www.tensorflow.org/)", "A powerful open-source library for numerical computation and large-scale machine learning."),
    ("Google Colab", "[Google Colab](https://colab.google/)", "A free Jupyter notebook environment that runs in the cloud."),
    ("PIL (Python Imaging Library)", "[PIL](https://pillow.readthedocs.io/)", "A popular library for working with images in Python."),
    ("Matplotlib", "[Matplotlib](https://matplotlib.org/)", "A comprehensive library for creating static, animated, and interactive visualizations in Python."),
    ("NumPy (Numerical Python)", "[NumPy](https://numpy.org/)", "A fundamental library for scientific computing in Python."),
    ("Streamlit", "[Streamlit](https://docs.streamlit.io/)", "A user-friendly library for creating web applications in Python."),
    ("cv2 (OpenCV)", "[OpenCV](https://opencv.org/)", "A powerful library for computer vision tasks like image processing, object detection, and video analysis."),
    ("time", "[Built-in Python module](https://docs.python.org/3/library/time.html)", "Provides functionalities for working with time and dates."),
    ("os", "[Built-in Python module](https://docs.python.org/3/library/os.html)", "Provides functionalities for interacting with the operating system."),
    ("imghdr", "[Built-in Python module](https://docs.python.org/3/library/imghdr.html)", "Helps determine the image format of a file based on its header information."),
]

    for name, link, description in libraries:
        st.subheader(name)
        st.markdown(f"{link} - {description}")
        st.write("---")

with tab2:
    st.subheader("Imports")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Dataset Preparation")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
    
    st.subheader("Preprocessing")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Model")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
    
    st.subheader("Model Summary")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Training step")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Plotting the Performance")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
        image = Image.open("perplot.png")
        st.image(image, use_column_width=True)

with tab3:

    st.subheader("Metrics used:")
    st.markdown("##### *Precision*")
    st.write("High precision indicates a low rate of false positives.")
    st.markdown("##### *Recall*")
    st.write("High recall indicates that the classes are correctly recognized.")
    st.markdown("##### *BinaryAccuracy*")
    st.write("Measures the percentage of correct predictions out of all predictions made.")

    st.divider()

    st.subheader("Evaluating the Performance")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')
    
    st.subheader("Output")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, language='python')

with tab4:
    st.subheader("Testing the model")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

with tab5:
    st.subheader("Home Page")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("About Page")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

    st.subheader("Source Code Page")
    code = "PLACEHOLDER_FOR_CODE"
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

'''
    with st.expander("Expand Code"):
        st.code(code, line_numbers=True, language='python')

