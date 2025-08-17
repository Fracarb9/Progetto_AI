import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.transform import resize
from skimage.util import img_as_ubyte



"""Fase 2: Preprocessing ed estrazione delle feature """
def extract_features(image):
    # Ridimensiona l'immagine a 128x128 (standardizzazione dimensioni)
    image = resize(image, (128, 128))
    # Conversione in uint8 (necessaria per LBP e GLCM)
    image_uint8 = (image * 255).astype(np.uint8)
    image = img_as_ubyte(image)

    # HOG
    hog_features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")

    # LBP
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    # Istogramma normalizzato delle frequenze dei pattern
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    #GLCM
    glcm = graycomatrix(image_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    # Proprietà mediate sui diversi angoli
    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    glcm_features = np.array([contrast, correlation, energy])

    # Concatenazione finale
    return np.concatenate((hog_features, lbp_hist, glcm_features))
