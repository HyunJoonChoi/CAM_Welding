import argparse
import cv2

from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import *
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from data import *
from model import *
from utils.image_utils import *


def train(train_dataset_path, val_dataset_path, epochs, batch_size, transfer_weights=False):
    model = get_cam_model(transfer_weights)

    cb_tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    checkpoint_path = "./weights/" + "original_vgg_16_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    cb_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, period=1, mode='auto')

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=30,
        zoom_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False)

    train_generator = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='categorical')

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        vertical_flip=False)

    val_generator = val_datagen.flow_from_directory(
        val_dataset_path,
        target_size=(200, 200),
        batch_size=batch_size,
        class_mode='categorical')

    print("Training..")
    model.fit_generator(
        train_generator,
        validation_data=val_generator,
        epochs=epochs, verbose=1,
        callbacks=[cb_tensorboard, cb_checkpoint])

def visualize_cam(model_path, image_samples, nb_samples):
    model = load_model(model_path)
    image_samples.astype(np.float32)

    cam_list = []; bbox_list = []
    for j in range(nb_samples):
        original_img = image_samples[j]
        width, height, _ = original_img.shape

        # Reshape to the network input shape (3, w, h).
        img = preprocess_input(np.expand_dims(original_img, axis=0).astype(np.float32))

        # Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = get_output_layer(model, "activation_49")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]
        predicted_label = np.argmax(predictions)

        # Create the class activation map.
        cam = np.zeros(shape=conv_outputs.shape[0:2], dtype=np.float32)
        for i, w in enumerate(class_weights[:, predicted_label]):
            cam += w * conv_outputs[:, :, i]

        print("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        # cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # pdb.set_trace()
        # plt.imshow(cam * 255); plt.show()

        cam[np.where(cam < 0.2)] = 0
        cam_list.append(cam)
        bbox = find_location_by_cam(cam, thresh=0.8)
        bbox_list.append(bbox)

    visualize(image_samples, cam_list, bbox_list, nb_samples)

def get_random_sample(samples):
    [X_OK, X_NG] = samples
    np.random.shuffle(X_OK); np.random.shuffle(X_NG)
    return (X_OK, X_NG)

if __name__ == '__main__':
    mode = 'test'
    transfer_weights   = True
    train_dataset_path = "Dataset/train"
    valid_dataset_path = "Dataset/valid"
    weight_path        = "./weights/" + "original_vgg_16_weights.10-0.05.hdf5"
    pickle_path        = './Dataset/valid/dataset.pickle'

    if mode == 'train':
        train(train_dataset_path, valid_dataset_path, epochs=10, batch_size=32, transfer_weights=True)
    elif mode == 'test':
        while (True):
            (X_OK, X_NG) = load_from_pickle(pickle_path)
            (X_OK, X_NG) = get_random_sample([X_OK, X_NG])            # O : 0, X : 1
            visualize_cam(weight_path, X_NG, nb_samples=2)
    else:
        raise NotImplementedError