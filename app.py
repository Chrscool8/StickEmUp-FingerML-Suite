import os.path
import numpy as np
import cv2
import os
import datetime
import sys
import glob
import PIL

# Modes
mode_main_menu = 0
mode_options = 1
mode_capture = 2
mode_testing = 3
mode_training = 4
mode_live = 5
mode = mode_main_menu

# Global Settings
window_width = 1280
window_height = 720
program_running = True
program_name = "StickEmUp"
working_directory = os.getcwd().replace("\\", "/")
training_directory = working_directory+"/images/training/"
testing_directory = working_directory+"/images/testing/"
header_height = 55

# Camera Settings
camera_initialized = False
camera_id = 0
camera_width = 1280
camera_height = 720
camera_handle = None

# Capturing Settings
target_x = int(window_width/2)
target_y = int(window_height/2)

# Mats
mat_display = None

# Color Constants
c_black = (0, 0, 0)
c_white = (255, 255, 255)
c_blue = (255, 0, 0)
c_red = (0, 0, 255)
c_green = (0, 255, 0)
c_teal = (255, 255, 0)
c_yellow = (0, 255, 255)
c_purple = (255, 0, 255)

c_theme_blue_darker = (163, 124, 56)
c_theme_blue_lighter = (200, 173, 82)
c_theme_blue_accent = (158, 195, 85)

# Keyboard
key = -1
key_escape = 27

# Click
click_happened = False
click_x = -1
click_y = -1

# Display Message
display_message = ""
display_message_timer = datetime.datetime.now()

# Textbox
textbox_training = []
textbox_testing = []
textbox_maxlines = 12

# TF
tensorflow_initialized = False

# Training
training_batch_size = 16
training_img_width = 128
training_img_height = 128
training_epochs = 10
training_in_action = False

# Testing
testing_in_action = False

# Model Info
model = None
model_path = "saved_model.h5"


def initialize_program():
    global mat_display, key
    mat_display = np.zeros((window_height, window_width, 3), np.uint8)

    print("Working Dir:", working_directory)
    print("Python:", sys.version)
    print("OpenCV:", cv2.__version__)
    #print("TensorFlow:", tf.__version__)

    for i in range(6):
        path_name = training_directory+"/hand_"+str(i)+"/"
        if not os.path.exists(path_name):
            os.makedirs(path_name)
    for i in range(6):
        path_name = training_directory+"/hand_"+str(i)+"/"
        if not os.path.exists(path_name):
            os.makedirs(path_name)


def init_tensorflow():
    global tensorflow_initialized

    if not tensorflow_initialized:
        tensorflow_initialized = True
        print("Initializing TF...")
        fill_with_color(img=mat_display, color=c_black)
        draw_header(subtitle="Loading TensorFlow, Please Wait...")
        draw_footer(subtitle="This takes a few seconds each time you run the program.")
        cv2.imshow(winname=program_name, mat=mat_display)
        cv2.waitKey(1)

    import tensorflow as tf
    print("TensorFlow:", tf.__version__)

    return tf


def initialize_training():
    textbox_training.clear()

    for i in range(6):
        file_count = str(len(list(glob.glob(training_directory+"/hand_"+str(i)+"/*.png"))))
        print_to_textbox(textbox=textbox_training, text="Hand "+str(i)+" has ["+str(file_count)+"] images.")

    print_to_textbox(textbox=textbox_training, text="Batch Size is ["+str(training_batch_size)+"]")
    print_to_textbox(textbox=textbox_training, text="Texture Size is ["+str(training_img_width)+","+str(training_img_height)+"]")
    print_to_textbox(textbox=textbox_training, text="Num of Epochs is ["+str(training_epochs)+"]")


def initialize_testing():
    global model
    textbox_testing.clear()

    print_to_textbox(textbox=textbox_testing, text="Loading Model: "+working_directory+"/"+model_path)
    tf = init_tensorflow()

    try:
        model = tf.keras.models.load_model(model_path)
        print_to_textbox(textbox=textbox_testing, text="Loaded model!")
    except:
        print_to_textbox(textbox=textbox_testing, text="Couldn't load model.")
        model = None


def initialize_live():
    global model
    tf = init_tensorflow()
    try:
        model = tf.keras.models.load_model(model_path)
        print_to_textbox(textbox=textbox_testing, text="Loaded model!")
    except:
        print_to_textbox(textbox=textbox_testing, text="Couldn't load model.")
        model = None


def show_notification(text):
    global display_message, display_message_timer
    display_message = "> " + text
    display_message_timer = datetime.datetime.now()


def draw_text(img, text, x, y, color, scale=1, font=cv2.FONT_HERSHEY_DUPLEX):
    default_text_height = 25
    cv2.putText(img=img, text=text, org=(int(x+2), int(y+default_text_height+2)),
                fontFace=font, fontScale=scale, color=(color[0]/4, color[1]/4, color[2]/4))
    cv2.putText(img=img, text=text, org=(int(x), int(y+default_text_height)), fontFace=font, fontScale=scale, color=color)


def draw_header(subtitle):
    global mat_display

    subtitle = "|  "+subtitle
    cv2.rectangle(img=mat_display, pt1=(0, 0), pt2=(window_width, header_height+2),
                  color=(c_theme_blue_darker[0]/4, c_theme_blue_darker[1]/4, c_theme_blue_darker[2]/4), thickness=-1)
    cv2.rectangle(img=mat_display, pt1=(0, 0), pt2=(window_width, header_height), color=c_theme_blue_darker, thickness=-1)
    draw_text(img=mat_display, text="StickEmUp: Finger Counting ML Demo", x=15, y=15, color=c_white)
    (subtitle_width, subtitle_height), baseline = cv2.getTextSize(text=subtitle, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
    draw_text(img=mat_display, text=subtitle, x=window_width-subtitle_width-15, y=15, color=c_white)


def draw_footer(subtitle):
    global mat_display

    footer_font = cv2.FONT_HERSHEY_PLAIN
    (subtitle_width, subtitle_height), baseline = cv2.getTextSize(text=subtitle, fontFace=footer_font, fontScale=1, thickness=1)
    cv2.rectangle(img=mat_display, pt1=(0, window_height-26), pt2=(window_width, window_height), color=c_theme_blue_darker, thickness=-1)
    draw_text(img=mat_display, text=subtitle, x=15, y=window_height - subtitle_height - 22, color=c_white, font=footer_font)


def draw_crosshair(mat, x, y, color):
    crosshair_size = 5

    cv2.circle(img=mat, center=(x, y), radius=crosshair_size, color=color)
    cv2.line(img=mat, pt1=(x, y-crosshair_size), pt2=(x, y+crosshair_size), color=color)
    cv2.line(img=mat, pt1=(x-crosshair_size, y), pt2=(x+crosshair_size, y), color=color)


def draw_notification():
    global mat_display, display_message, display_message_timer
    now = datetime.datetime.now()
    if (now - display_message_timer).seconds < 5:
        (subtitle_width, subtitle_height), baseline = cv2.getTextSize(text=display_message, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
        draw_text(img=mat_display, text=display_message, x=window_width-subtitle_width - 15, y=header_height+30, color=c_white, scale=.75)


def fill_with_color(img, color):
    img[0:img.shape[0], 0:img.shape[1]] = color


def capture_hand():  # Returns: (Success, Mat, TopLeft (x, y), BottomRight (x, y))
    global click_happened, click_x, click_y, target_x, target_y, key
    mat_camera = get_camera_frame()

    mat_camera = cv2.cvtColor(src=mat_camera, code=cv2.COLOR_BGR2HSV)
    mat_camera_hue, mat_camera_sat, mat_camera_val = cv2.split(mat_camera)

    retval, mat_camera_val = cv2.threshold(src=mat_camera_val, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

    mat_camera_val = cv2.dilate(src=mat_camera_val, kernel=cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5)))
    #mat_camera_val = cv2.erode(src=mat_camera_val, kernel=cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3)))
    mat_camera_val = cv2.merge([mat_camera_val, mat_camera_val, mat_camera_val])

    cv2.floodFill(image=mat_camera_val, mask=None, seedPoint=(target_x, target_y), newVal=c_blue)
    mat_camera_val[np.where((mat_camera_val == [255, 255, 255]).all(axis=2))] = c_black
    mat_camera_val[np.where((mat_camera_val == c_blue).all(axis=2))] = c_white

    # For filling in gaps
    #mat_inpaint = mat_camera_val.copy()
    #cv2.floodFill(image=mat_inpaint, mask=None, seedPoint=(0, 0), newVal=c_white)
    #mat_inpaint = (255-mat_inpaint)
    #mat_camera_val += mat_inpaint

    min_x, min_y, width, height = cv2.boundingRect(array=cv2.split(mat_camera_val)[0])
    print(min_x, min_y, width, height)

    max_x = min_x+width
    max_y = min_y+height

    mid_x = (min_x + max_x)/2
    mid_y = (min_y + max_y)/2

    size = max(width, height)

    if (size > 5 and size < min(window_width, window_height)*.9):
        return True, mat_camera_val[int(mid_y-size/2):int(mid_y+size/2), int(mid_x-size/2):int(mid_x+size/2)], (int(mid_x-size/2), int(mid_y-size/2)), (int(mid_x+size/2), int(mid_y+size/2))

    return False, None, (0, 0), (0, 0)


def save_iteration(dir, name, mat):
    print(dir+name+"*")
    fileticker = 0
    while(fileticker < 100000):
        fileticker += 1
        full_path = dir+name+str(fileticker)+".png"
        if not os.path.exists(full_path):
            cv2.imwrite(full_path, mat)
            show_notification(text="Saved "+full_path)
            break


def click_mouse(event, x, y, flags, param):
    global click_happened, click_x, click_y

    if event == cv2.EVENT_LBUTTONDOWN:
        click_happened = True
        click_x = x
        click_y = y
        print('CLICK: x = '+str(x)+', y = '+str(y))


def reset_mouse():
    global click_happened, click_x, click_y

    click_happened = False
    click_x = -1
    click_y = -1


def get_camera_frame():
    global camera_id, camera_initialized, camera_handle

    if not camera_initialized:
        print("Initializing Camera...")
        fill_with_color(img=mat_display, color=c_black)
        draw_header(subtitle="Loading Camera, Please Wait...")
        draw_footer(subtitle="This takes a few seconds each time you run the program.")
        cv2.imshow(winname=program_name, mat=mat_display)
        cv2.waitKey(1)
        camera_initialized = True
        camera_handle = cv2.VideoCapture(camera_id)
        camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        print("Found Webcam!")

    ret_val, mat_camera = camera_handle.read()

    if not ret_val:
        mat_camera = np.zeros((camera_height, camera_width, 3), np.uint8)

    return mat_camera


def print_to_textbox(textbox, text):
    textbox.append("- "+text)
    print(text)
    if len(textbox) > textbox_maxlines:
        textbox.pop(0)


def run_main_menu():
    global key, program_running, mode

    fill_with_color(img=mat_display, color=c_theme_blue_lighter)
    draw_header(subtitle="Main Menu")
    draw_footer(subtitle="1-6: Select Menu Item")

    draw_text(img=mat_display, text="1) Capture", x=30, y=85, color=c_white)
    draw_text(img=mat_display, text="2) Train", x=30, y=140, color=c_white)
    draw_text(img=mat_display, text="3) Test", x=30, y=195, color=c_white)
    draw_text(img=mat_display, text="4) Live Test", x=30, y=250, color=c_white)
    draw_text(img=mat_display, text="5) Options and Diags", x=30, y=305, color=c_white)
    draw_text(img=mat_display, text="6) Quit", x=30, y=360, color=c_white)

    if key == ord('1'):
        mode = mode_capture
    elif key == ord('2'):
        initialize_training()
        mode = mode_training
    elif key == ord('3'):
        initialize_testing()
        mode = mode_testing
    elif key == ord('4'):
        initialize_live()
        mode = mode_live
    elif key == ord('5'):
        mode = mode_options
    elif key == ord('6'):
        program_running = False


def run_options():
    global mat_display
    fill_with_color(img=mat_display, color=(128, 64, 128))
    draw_header(subtitle="Options")
    draw_footer(subtitle="Escape: Back to Main Menu")

    camera_preview_w = int(window_width/4)
    camera_preview_h = int(window_height/4)

    camera_preview = cv2.resize(src=get_camera_frame(), dsize=(camera_preview_w, camera_preview_h))

    for _x in range(camera_preview_w):
        for _y in range(camera_preview_h):
            mat_display[window_height-camera_preview_h+_y, window_width-camera_preview_w+_x] = camera_preview[_y, _x]


def run_capture():
    global mat_display, click_happened, click_x, click_y, target_x, target_y, key

    fill_with_color(img=mat_display, color=(64, 128, 128))

    mat_camera = cv2.resize(src=get_camera_frame(), dsize=(window_width, window_height))
    mat_display = mat_camera.copy()

    if click_happened:
        target_x = int(click_x/window_width*camera_width)
        target_y = int(click_y/window_height*camera_height)
        print(target_x, target_y)

    if key >= ord('0') and key <= ord('5'):
        success, mat_hand, min_coords, max_coords = capture_hand()
        if (success):
            # cv2.imshow(winname="Hand", mat=mat_hand)
            number = str(key-ord('0'))
            save_iteration(dir=training_directory+"hand_"+number+"/", name="hand_"+number+"-", mat=mat_hand)
        else:
            show_notification(text="Couldn't capture hand")

    draw_crosshair(mat=mat_display, x=target_x, y=target_y, color=c_red)

    draw_header(subtitle="Capture")
    draw_footer(subtitle="Escape: Back to Main Menu  |  Click: Set Origin  |  0-5: Capture Frame")


def run_testing():
    global mat_display, testing_in_action
    fill_with_color(img=mat_display, color=(128, 128, 64))
    draw_header(subtitle="Testing")

    yy = header_height+20
    for t in textbox_testing:
        draw_text(img=mat_display, text=t, x=30, y=yy, color=c_white, scale=.75)
        yy += 50*.75

    if testing_in_action:
        draw_footer(subtitle="Please wait while the model tests!")
        cv2.imshow(winname=program_name, mat=mat_display)
        cv2.waitKey(1)
    else:
        draw_footer(subtitle="Enter: Begin Testing Session")

    if not testing_in_action and key == 13:  # [ Enter] Key
        testing_in_action = True
        tf = init_tensorflow()

        results_list = []

        class_names = glob.glob(pathname=testing_directory+"*/", recursive=False)
        for i in range(len(class_names)):
            class_names[i] = os.path.split(class_names[i].replace("\\", "/")[:-1])[-1]

        for expected in class_names:
            number_tested = 0
            number_correct = 0

            dir_name = testing_directory+expected+"/"

            list_of_files = filter(os.path.isfile, glob.glob(dir_name + '*.png'))

            for file_path in list_of_files:
                img = tf.keras.utils.load_img(file_path, target_size=(training_img_height, training_img_width))

                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                winner = np.argmax(score)

                print_to_textbox(textbox=textbox_testing, text="{} most likely belongs to {} with a {:.2f} percent confidence.".format(
                    os.path.basename(file_path), class_names[winner], 100 * np.max(score)))
                run_testing()

                number_tested += 1
                if class_names[winner] == expected:
                    number_correct += 1

            print_to_textbox(textbox=textbox_testing, text="Tested: "+str(number_tested))
            print_to_textbox(textbox=textbox_testing, text="Correct: "+str(number_correct))
            print_to_textbox(textbox=textbox_testing, text="Percent: "+str(number_correct/number_tested*100)+"%")
            run_testing()

            results_list.append([expected, (str(number_correct/number_tested*100)+"%")])

        for r in results_list:
            print_to_textbox(textbox=textbox_testing, text=str(r))

        testing_in_action = False


def run_training():
    global mat_display, textbox_training, training_in_action

    fill_with_color(img=mat_display, color=(128, 128, 64))
    draw_header(subtitle="Training")

    yy = header_height+20
    for t in textbox_training:
        draw_text(img=mat_display, text=t, x=30, y=yy, color=c_white)
        yy += 50

    if training_in_action:
        draw_footer(subtitle="Please wait while the model trains!")
        cv2.imshow(winname=program_name, mat=mat_display)
        cv2.waitKey(1)
    else:
        draw_footer(subtitle="Enter: Begin Training Session")

    if not training_in_action and key == 13:  # [ Enter] Key
        training_in_action = True
        tf = init_tensorflow()

        # All the built-in callbacks, grabbing the ones I want to use for now
        class AllCallbacks(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                print_to_textbox(textbox=textbox_training, text="- - - - - - - -")
                print_to_textbox(textbox=textbox_training, text="Starting training")
                run_training()

            def on_train_end(self, logs=None):
                keys = list(logs.keys())
                print_to_textbox(textbox=textbox_training, text="Training complete!")
                run_training()

            def on_epoch_begin(self, epoch, logs=None):
                keys = list(logs.keys())
                print_to_textbox(textbox=textbox_training, text="Starting epoch "+str(epoch)+"/"+str(training_epochs))
                run_training()

            def on_epoch_end(self, epoch, logs=None):
                keys = list(logs.keys())
                print_to_textbox(textbox=textbox_training, text="Finished epoch "+str(epoch)+"/"+str(training_epochs))
                run_training()

            def on_test_begin(self, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="Start testing; got log keys: {}".format(keys))
                run_training()

            def on_test_end(self, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="Stop testing; got log keys: {}".format(keys))
                run_training()

            def on_predict_begin(self, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="Start predicting; got log keys: {}".format(keys))
                run_training()

            def on_predict_end(self, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="Stop predicting; got log keys: {}".format(keys))
                run_training()

            def on_train_batch_begin(self, batch, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="...Training: start of batch {}; got log keys: {}".format(batch, keys))
                run_training()

            def on_train_batch_end(self, batch, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="...Training: end of batch {}; got log keys: {}".format(batch, keys))
                run_training()

            def on_test_batch_begin(self, batch, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))
                run_training()

            def on_test_batch_end(self, batch, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))
                run_training()

            def on_predict_batch_begin(self, batch, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="...Predicting: start of batch {}; got log keys: {}".format(batch, keys))
                run_training()

            def on_predict_batch_end(self, batch, logs=None):
                keys = list(logs.keys())
                #print_to_textbox(textbox=textbox_training, text="...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
                run_training()

        train_ds = tf.keras.utils.image_dataset_from_directory(
            training_directory,
            subset="training",
            validation_split=0.2,
            seed=8,
            image_size=(training_img_height, training_img_width),
            batch_size=training_batch_size)

        val_ds = tf.keras.utils.image_dataset_from_directory(
            training_directory,
            validation_split=0.2,
            subset="validation",
            seed=8,
            image_size=(training_img_height, training_img_width),
            batch_size=training_batch_size)

        class_names = train_ds.class_names
        num_classes = len(class_names)

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1./255, input_shape=(training_img_height, training_img_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        model.summary()

        history = model.fit(train_ds, validation_data=val_ds, epochs=training_epochs, callbacks=[AllCallbacks()])

        model.save(model_path)
        print_to_textbox(textbox=textbox_training, text="Saved model: "+working_directory+"/"+model_path)

        training_in_action = False


def run_live():
    global mat_display, click_happened, click_x, click_y, target_x, target_y, key, model
    fill_with_color(img=mat_display, color=(64, 128, 64))
    mat_camera = cv2.resize(src=get_camera_frame(), dsize=(window_width, window_height))
    mat_display = mat_camera.copy()

    if click_happened:
        target_x = int(click_x/window_width*camera_width)
        target_y = int(click_y/window_height*camera_height)
        print(target_x, target_y)

    success, mat_hand, min_coords, max_coords = capture_hand()
    if (success):
        if (max_coords[0]-min_coords[0] > 5 and max_coords[1]-min_coords[1] > 5):
            cv2.rectangle(img=mat_display, pt1=min_coords, pt2=max_coords, color=c_blue)
            mat_display[min_coords[1]: max_coords[1], min_coords[0]: max_coords[0]] = mat_hand

            tf = init_tensorflow()

            mat_hand = cv2.cvtColor(mat_hand, cv2.COLOR_BGR2RGB)
            mat_hand = cv2.resize(mat_hand, (training_img_height, training_img_width), interpolation=cv2.INTER_NEAREST)
            pil_hand = PIL.Image.fromarray(mat_hand)

            img_array = tf.keras.utils.img_to_array(pil_hand)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            class_names = glob.glob(pathname=testing_directory+"*/", recursive=False)
            for i in range(len(class_names)):
                class_names[i] = os.path.split(class_names[i].replace("\\", "/")[:-1])[-1]

            winner = np.argmax(score)

            show_notification(text="This most likely belongs to {} with a {:.2f} percent confidence.".format(
                class_names[winner], 100 * np.max(score)))

    else:
        show_notification(text="Couldn't capture hand")

    draw_crosshair(mat=mat_display, x=target_x, y=target_y, color=c_red)

    draw_header(subtitle="Live Test")
    draw_footer(subtitle="Escape: Back to Main Menu  |  Click: Set Origin")


def main():
    global mat_display, key, program_running, mode, training_in_action
    initialize_program()

    cv2.imshow(winname=program_name, mat=mat_display)

    while(program_running):
        key = cv2.waitKeyEx(delay=1)
        if key == key_escape and not training_in_action and not testing_in_action:
            mode = mode_main_menu

        if key != -1:
            print(key)

        if key == ord('b'):
            show_notification("Bloop")

        if mode == mode_main_menu:
            run_main_menu()
        elif mode == mode_options:
            run_options()
        elif mode == mode_capture:
            run_capture()
        elif mode == mode_testing:
            run_testing()
        elif mode == mode_training:
            run_training()
        elif mode == mode_live:
            run_live()

        if cv2.getWindowProperty(winname=program_name, prop_id=cv2.WND_PROP_VISIBLE) < 1:
            break

        draw_notification()

        reset_mouse()

        cv2.imshow(winname=program_name, mat=mat_display)
        cv2.setMouseCallback(program_name, click_mouse)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
