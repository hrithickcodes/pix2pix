import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime


# Paths
folderPath = "data/dataset/train"
sourceNPYPath = "data/preprocessed_data/source.npy"
targetNPYPath = "data/preprocessed_data/target.npy"


def separate_source_target(image):
	"""
	Input: One image of shape (600, 1200, 3)
	Returns: Two image of shape (600, 600, 3), (600, 600, 3)
	"""

	source = image[:, 0: 600, :]
	target = image[:, 600: 1200, :]
	return source, target


def preprocess_and_save(folder_path, source_saver_path, target_saver_path):
	"""
	Input: Training data folder path
	Output: Saves source and target images, both as .npy format
	"""
	start_time = datetime.now()

	source_tensor, target_tensor = [], []
	# all image names in the given folder_path
	image_names_list = os.listdir(folder_path)


	for image in tqdm(image_names_list):
		# path of the image
		source_target_image_path = os.path.join(folder_path, image)
		# reading the image using opencv, bydefault in BGR format
		map_image = cv2.imread(source_target_image_path)
		# BGR -> RGB
		RGB_map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

		source_map, target_map = separate_source_target(image = RGB_map_image)
		# resizing both of the images
		source_resized = cv2.resize(source_map,
								   (256, 256),
								   interpolation = cv2.INTER_NEAREST)
		target_resized = cv2.resize(target_map,
								   (256, 256),
								   interpolation = cv2.INTER_NEAREST)

		# appending both of the images 
		source_tensor.append(source_resized)
		target_tensor.append(target_resized)


	source_tensor = np.array(source_tensor)
	target_tensor = np.array(target_tensor)

	# saving as NPY files
	np.save(source_saver_path, source_tensor)
	np.save(target_saver_path, target_tensor)

	end_time = datetime.now()

	print("\n")
	print('Time took: {}'.format(end_time - start_time))





preprocess_and_save(folder_path = folderPath,
					source_saver_path = sourceNPYPath,
					target_saver_path = targetNPYPath)