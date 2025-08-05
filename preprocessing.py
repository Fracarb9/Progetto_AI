import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os


def load_dataset(folder_path, img_size=(128, 128), verbose=True):
    images = []
    labels = []
    label_map = {'healthy': 0, 'parkinson': 1}

    for label_name in sorted(label_map.keys()):
        class_dir = os.path.join(folder_path, label_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith('.jpg'):
                continue
            img_path = os.path.join(class_dir, file_name)
            try:
                img = imread(img_path, as_gray=True)  # grayscale
                if img_size:
                    img = resize(img, img_size, anti_aliasing=True)
                images.append(img)
                labels.append(label_map[label_name])
            except Exception as e:
                print(f"Errore caricando {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    if verbose:
        print(f"Dataset caricato da: {folder_path}")
        print(f"Totale immagini: {len(images)}")
        print(f"Distribuzione classi: healthy={np.sum(labels == 0)}, parkinson={np.sum(labels == 1)}")

    return images, labels
