from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np
import pickle
import pdb
from random import shuffle


def load_images(dataset_dir, nb_samples, target_size):
  dir_list = categories = ['OK', 'NG']
  nb_classes = len(categories)

  X_OK = []; X_NG = []
  (image_w, image_h) = target_size

  for idx, cls in enumerate(dir_list):
    print("%s : %d" %(cls, idx))

  for idx, f in enumerate(dir_list):
    label = [0 for _ in range(nb_classes)]
    label[idx] = 1

    dir = dataset_dir + "/" + f
    imgs = glob.glob(dir + "/*.jpg")[:nb_samples[idx]]
    shuffle(imgs)

    for i, fname in enumerate(imgs):
      img = Image.open(fname)
      img = img.convert("RGB")
      img = img.resize((image_w, image_h))
      data = np.asarray(img)

      if f == 'OK':
        X_OK.append(data)
      else:
        X_NG.append(data)

  return (np.asarray(X_OK), np.asarray(X_NG))

def save_to_pickle(data, path):
  '''
  :param data: dict including train and test dataset
  :param path: path to save pickle file
  '''

  if os.path.isfile(path):
    os.remove(path)
  with open(path, 'wb') as f:
    pickle.dump(data, f)

def load_from_pickle(path):
  with open(path, 'rb') as f:
    output = pickle.load(f)
  X_OK = output['X_OK']
  X_NG = output['X_NG']

  return (X_OK, X_NG)


# Execute
if __name__ == '__main__':
  dataset_dir = './Dataset/valid'
  pickle_path = './Dataset/valid/dataset.pickle'

  (X_OK, X_NG) = load_images(dataset_dir,
                             [200, 100],
                             target_size=(200, 200))
  dataset = {'X_OK': X_OK, 'X_NG': X_NG}
  save_to_pickle(dataset, pickle_path)

  # (X_OK, X_NG) = load_from_pickle(pickle_path)
  #
  # pdb.set_trace()