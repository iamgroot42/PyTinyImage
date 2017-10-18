import numpy as np
from tqdm import tqdm as progressbar
import tinyimage


def get_labels():
	labels = []
	with open('./labels_cifar100','r') as f:
		for label in f:
			labels.append(label.rstrip())
	return labels


def get_indices():
	indices = []
	with open('./indices_cifar100','r') as f:
		for index in f:
			indices.append(int(index.rstrip()))
	return indices


if __name__ == "__main__":
	keywords = get_labels()
	tinyimage.openTinyImage()
	images = []
	ignore_indices = get_indices()
	pick = len(ignore_indices)
	for keyword in progressbar(keywords):
		indexes = tinyimage.retrieveByTerm(keyword)
		for i in indexes:
			if i not in ignore_indices:
				image = tinyimage.sliceToBin(i).reshape(32,32,3, order="F").astype('float32') / 255.
				images.append(image)
	relevant = np.array(images)
	np.random.shuffle(relevant)
	relevant = relevant[:pick]
	np.save("relevant_images",relevant)
	tinyimage.closeTinyImage()
