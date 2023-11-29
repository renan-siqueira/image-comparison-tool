from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics.pairwise import euclidean_distances
from src.settings import config


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, feature_vector=True)
    return features


def compare_images(image1_features, image2_features):
    return euclidean_distances([image1_features], [image2_features]).min()


def main(dataset_dir, generated_dir, log_file):
    original_images = [f for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]
    generated_images = [f for f in listdir(generated_dir) if isfile(join(generated_dir, f))]

    with open(log_file, 'w') as log:
        for gen_image in generated_images:
            gen_features = extract_features(join(generated_dir, gen_image))
            distances = []

            for orig_image in original_images:
                orig_features = extract_features(join(dataset_dir, orig_image))
                distance = compare_images(gen_features, orig_features)
                distances.append((orig_image, distance))

            distances.sort(key=lambda x: x[1])

            log.write(f"Generated Image: {gen_image}\n")
            for orig_image, distance in distances:
                log.write(f"\t{orig_image} - Distance: {distance:.2f}\n")
            log.write("\n")


if __name__ == "__main__":
    ORIGINAL_IMAGES = config.APP_PATH_ORIGINAL_IMAGES
    GENERATED_IMAGES = config.APP_PATH_GENERATED_IMAGES
    LOG_FILE = config.APP_LOG_FILENAME

    main(ORIGINAL_IMAGES, GENERATED_IMAGES, LOG_FILE)
