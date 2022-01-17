# Some helper functions for tensroflow projects


## ------------- DATA PREPROCESSING --------------------

def view_images(target_dir,class_names):
    """
    Plots out a random image for each class available 
    """
    plt.figure(figsize=(20,20))
    for i in range(len(class_names)):
        plt.subplot(5,5,i + 1)

        target_folder = target_dir + "/" + class_names[i]
        random_image = random.sample(os.listdir(target_folder),1) # Selects a random image file
        img = mpimg.imread(target_folder + "/" + random_image[0]) # Gets the image  
        # Plot the image and the image info
        plt.imshow(img)
        plt.axis("off")
        image_info = class_name[i] + "\n" + str(img.shape)
        plt.title(image_info)


def unzip_data(filename):
    """
    Unzips filename into the current directory
    """
    zip_ref = zipfile.Zipfile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


def walk_through_dir(file_path):
    """
    Walks through the the file_path
    and prints the subfolders and images
    """
    for dirpath,dirnames ,filenames in os.walk(file_path):
        if len(dirnames == 0):
            print(f"There are {len(filenames)} images inside {dirpath}")
        else:
            print(f"There are {dirnames} directories inside {dirpath}")


def get_class_names(file_path):
    """
    Gets the class names from inside the file_path

    Returns:
        The class names from file_path
    """
    class_names = np.array(sorted(item.name for item in data_dir.glob("*")))
    print(f"The class names are: \n {class_names}")
    return class_names



## ------------- FUNCTIONS FOR MODEL--------------------
def create_tfhub_model(model_url,num_classes,image_shape=(224,224),activation_layer = "softmax"):
    """
    Creates a keras model from a tf hub url

    Args:
        model_url: Valid url from tensorflow hub
        num_classes: Length of classes
        image_shape: The shape that you want the images in.
        (224,224) by default
        activation_layer: Activation layer for the model by default softmax
        Change to sigmoid if binary classes

    Returns:
        An uncomiled Keras Sequential model
    """
    # Download the pretained model and freeze its parameters
    feature_extractor_layer = hub.KerasLayer(model_url,
                                            trainable= False,
                                            name = "Feature_Extractor_Layer",
                                            input_shape = image_shape + (3,))
    
    # Create our own model using the feature extractor layer
    model = tf.keras.Sequential([
            feature_extractor_layer,
            layers.Dense(num_classes,activation = activation_layer,name = "Output_layer")
    ])

    return model



def create_tensorboard_callback(dir_name,experiment_name):
    """
    Creates a Tensorboar callback instance to store log files
    Stores log files with the filepath:
        'dir_name/experiment_name/current datetime' 
    Args:
        dir_name = target direcotry to store tensorboard log files
        experiment_name = name of experiment directory
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%d-%m-%Y|%H:%M")
    tensorboard_callback = tf.keras.callbacks.Tensorboard(
        log_dir = log_dir
    )
    print(f"Saving Tensorboard log files to: {log_dir}")
    return tensorboard_callback



## ------------- FUNCTIONS FOR AFTER THE MODEL HAS BEEN TRAINED --------------------

def plot_loss_curves(history,validation = True):
    """
    Plots out the loss and accuracy curves
    
    Args:
        history: Tensorflow model history
        validation: Boolean on either the model contains or not a validation set
    """   

    loss = history.history['loss']
    accuracy = history.history["accuracy"]

    if validation:
        val_loss = history.history["val_loss"]
        val_acc = history.history["val_accuracy"]
    epochs = range(len(history.history['accuracy']))

    # Plot out the loss
    plt.plot(epochs,loss,label="Training loss")
    if validation:
        plt.plot(epochs,val_loss,label = "Validation loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot out the accuracy
    plt.plot(epochs,accuracy,label="Training accuracy")
    if validation:
        plt.plot(epochs,val_acc,label="Validation accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def calculate_results(y_true,y_pred):
    """
    Calculates a models accuracy , precision , recall 
    and f1 score o 
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true,y_pred) * 100

    # Calculate model precision recall and f1 score using weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                    "precision": model_precision,
                    "recall": model_recall,
                    "f1": model_f1}
    return model_results


def load_and_prep_custom_images(filename,image_shape = 224,scale = True):
    """
    Reads an image from the filename and turns it into a tensor
    Reshapes the image into 224 by default
    Rescales the image into 0 and 1s

    Args:
        filename: Path to the directory containing images
        image_shape: Shape for the images , 
        by defaut 224,224
        scale: Boolean on whethe the images need to be scaled or not 
    """
    # Read the image
    img = tf.io.read_file(filename)
    # Decode img into a tensor
    img = tf.image_decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img,[image_shape,image_shape])
    
    if scale:
        return img/255.
    else:
        return img



def pred_and_plot(model,file_path,class_names):
    """
    Predicts and plots custom images from a directory 
    
    Args:
        model: Trained model which will make the predictions
        file_path: Path to the image directory
        class_names: The class names of the data used in training
    """
    # Setup figure size
    plt.figure(figsize=(20,20))

    # Get all the filenames under the file_path
    filenames = os.listdir(file_path)

    # Time to get the images from the path
    for i in range(len(filenames)):
        img = load_and_prep_custom_images(filename[i])
        # Make the model predict on the image
        pred = model.predict(tf.expand_dims(img,axis=0))

        # Get the predicted class
        if len(pred[0] > 1): # Check for multi class labels
            pred_class = class_names[pred.argmax()] # If multiple labels take the max
        else:
            pred_class = class_names[int(tf.round(pred[0][0]))] # Round if the classes are binary
        
        # PLot out the image
        plt.subplot(5,5,i+1)
        plt.imshow(img)
        plt.title(f"Prediction: \n {pred_class}")
        plt.axis("off")


# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")