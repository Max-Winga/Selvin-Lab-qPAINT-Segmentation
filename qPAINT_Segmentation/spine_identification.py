import tensorflow as tf
from tensorflow.keras.models import load_model
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import numpy as np
import tifffile
import json
import os


def load_life_act(life_act, movie_index=0):
    """
    Function to load life_act for the class from a file.

    Args:
        life_act (str): string path to the life act file
        movie_index (int): index of the tif movie containing the life_act background. Defaults to 0.

    Returns:
        np.ndarray: the first frame of the life_act movie
    """
    try:
        with tifffile.TiffFile(life_act) as tif:
            n_frames = len(tif.pages)
            movie = np.zeros((n_frames, tif.pages[0].shape[0], 
                                tif.pages[0].shape[1]), dtype='uint16')
            for i in range(n_frames):
                movie[i,:,:] = tif.pages[i].asarray()
        life_act = movie[movie_index]
    except:
        raise Exception(f"""Issues with path: {life_act}, could not load movie""")
    if not isinstance(life_act, np.ndarray):
        raise RuntimeError(f"life_act is of type: {type(life_act)}, must be a string to the filepath")
    return life_act

def locate_spines(model_path, life_act_path, input_shape, life_act_thresh, pred_thresh, life_act_movie_index=0):
    """Function to locate spines using DeepD3 and Stardist

    Args:
        model_path (string): file path to the deepd3 model to use.
        input_shape ((int, int)): shape to scale self.life_act to for processing.
        life_act_thresh (float): Threshold value for spines against the background.
        pred_thresh (float): Threshold value for predictions to count in range [0, 1].
        life_act_movie_index (int): Index of the tif movie containing the life_act background. Defaults to 0.

    Returns:
        Spines (list[Spine]): A dictionary containing Spine classes
        stardist (2D array): A 2D array of labels for spines. No spine = -1, else 0, 1, 2, ...
    """
    # Load model and background
    model = load_model(model_path, compile=False)
    life_act = load_life_act(life_act_path, life_act_movie_index)
    background = np.copy(life_act)
    normalized_background = 2 * (background / np.max(background)) - 1
    normalized_background = np.expand_dims(normalized_background, axis=-1)
    resized_background = np.expand_dims(tf.image.resize(normalized_background, input_shape), axis=0)
    
    # Make Predictions
    spine_predictions = model.predict(resized_background)[1]
    resized_preds = tf.image.resize(spine_predictions, life_act.shape).numpy().squeeze()
    bin_pred = resized_preds * (life_act > life_act_thresh) * (resized_preds > pred_thresh)
    normalized_predictions = normalize(bin_pred, 0, 99.8)
    
    # Use Stardist to classify predictions
    star_model = StarDist2D.from_pretrained('2D_versatile_fluo')
    starplane, _ = star_model.predict_instances(normalized_predictions, prob_thresh=0.3, nms_thresh=0.3)
    
    # Create a dictionary of pixels for each 2d stardist label
    label_dict = {}
    next_label_index = 0
    labels = []
    labels_roi = {}
    for y in range(len(starplane)):
        for x in range(len(starplane[0])):
            label = starplane[y][x]
            if label != 0:
                # Get the spine label
                if not label in label_dict:
                    label_dict[label] = next_label_index
                    next_label_index += 1
                spine_label = label_dict[label]

                # Update labels_roi
                if spine_label in labels:
                    labels_roi[spine_label].append([x,y])
                else:
                    labels.append(spine_label)
                    labels_roi[spine_label] = [[x,y]]
                
                # Update starplane
                starplane[y][x] = spine_label
            else:
                starplane[y][x] = -1
    return starplane, labels_roi

def save_spine_data(starplane, labels_roi, output_dir, filename):
    """
    Function to save the starplane and labels_roi data to separate files.

    Args:
        starplane (np.ndarray): 2D array of spine labels.
        labels_roi (dict): Dictionary mapping spine labels to ROI coordinates.
        output_dir (str): Directory to save the output files.
        filename (str): Base filename for the output files (without extension).

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the starplane as a numpy file
    starplane_file = os.path.join(output_dir, f"{filename}_starplane.npy")
    np.save(starplane_file, starplane)

    # Save the labels_roi as a JSON file
    labels_roi_file = os.path.join(output_dir, f"{filename}_labels_roi.json")
    with open(labels_roi_file, "w") as f:
        json.dump(labels_roi, f)

    print(f"Spine data saved successfully.")
    print(f"Starplane file: {starplane_file}")
    print(f"Labels ROI file: {labels_roi_file}")