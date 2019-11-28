import matplotlib

# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')

from keras.applications import mobilenetv2, inception_resnet_v2, nasnet
from keras.layers import Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, rmsprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import json

from src.data_loader.data_generator import SignDataLoader, get_data_for_master_class

classes = {
    # TODO: Add signs from belgium? change the class used?
    "RedRoundSign": {
        "signs_classes": ['p1', 'p10', 'p11', 'p12', 'p19', 'p20', 'p23', 'p26', 'p27', 'p3', 'p5L', 'p6', 'p9', 'pax',
                          'pb', 'phx', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70',
                          'pl80', 'pmx', 'p_prohibited_bicycle_and_pedestrian', 'p_prohibited_bus_and_truck',
                          'p_prohibited_other', 'prx', 'p_other', 'plo'],
        "merge_sign_classes": {
            "prx": ['pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'prx'],
            "pmx": ['pm1.5', 'pm10', 'pm13', 'pm15', 'pm2', 'pm2.5', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46',
                    'pm5', 'pm49', 'pm50', 'pm55', 'pm8'],
            "phx": ['ph', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3.x', 'ph3',
                    'ph3.2', 'ph3.3', 'ph3.5', 'ph3.7', 'ph3.8', 'ph38', 'ph39', 'ph45', 'ph4', 'ph4.2', 'ph4.3',
                    'ph4.4', 'ph4.5', 'ph4.6', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'ph6'],
            "pax": ['pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pax'],
            "plo": ['pl35', 'pl25', 'pl15', 'pl10', 'pl110', 'pl65', 'pl90'],
            "p_other": ['pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5',
                        'p_prohibited_two_wheels_vehicules', 'p_prohibited_bicycle_and_pedestria',
                        'p_prohibited_bicycle_and_pedestrian_issues', 'p13', 'p15', 'p16', 'p17', 'p18', 'p2', 'p21',
                        'p22', 'p24', 'p25', 'p28', 'p4', 'p5R', 'p7L', 'p7R', 'p8', 'p15', 'pc']
            },
        "h_symmetry": [],
        "rotation_and_flips": {  # "pne": ('v', 'h', 'd'),
            # "pn": ('v', 'h', 'd'),
            # "pnl": ('d',),
            # "pc": ('v', 'h', 'd'),
            "pb": ('v', 'h', 'd'),
            "p_other": ('v', 'h', 'd'),
        }
    },
}


def plot_history(history, base_name=""):
    plt.clf()
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(base_name + "accuracy.png")
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(base_name + "loss.png")
    plt.clf()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_class", type=str, choices=classes.keys())
    parser.add_argument("data_dir", type=str)
    parser.add_argument('-e', "--epoch",
                        required=False,
                        type=int,
                        default=64,
                        dest="epoch")
    parser.add_argument('-b', '--batch-size',
                        required=False,
                        default=1024,
                        type=int,
                        dest="batch")
    parser.add_argument('-lr', '--learning-rate',
                        required=False,
                        default=1e-4,
                        type=float,
                        dest="lr")
    parser.add_argument('-decay', '--learning-rate-decay',
                        required=False,
                        default=1e-6,
                        type=float,
                        dest="decay")
    parser.add_argument('-ignore-npz', '--ignore-precomputed-learning-file',
                        required=False,
                        default=False,
                        type=bool,
                        dest="ignore_npz")
    parser.add_argument('-is', '--input-size',
                        required=False,
                        default=96,
                        type=int,
                        dest="input_size")
    args = parser.parse_args()
    batch_size = args.batch

    class_name = args.train_class
    out_classes = classes[class_name]["signs_classes"]
    rotation_and_flips = classes[class_name]["rotation_and_flips"]
    h_symmetry_classes = classes[class_name]["h_symmetry"]
    try:
        merge_sign_classes = classes[class_name]["merge_sign_classes"]
    except KeyError:
        merge_sign_classes = None

    mapping = {c: i for i, c in enumerate(out_classes)}
    mapping_id_to_name = {i: c for c, i in mapping.items()}

    os.makedirs(class_name, exist_ok=True)

    x_train, y_train, x_test, y_test = get_data_for_master_class(class_name=class_name,
                                                                 mapping=mapping,
                                                                 mapping_id_to_name=mapping_id_to_name,
                                                                 rotation_and_flips=rotation_and_flips,
                                                                 data_dir=args.data_dir,
                                                                 merge_sign_classes=merge_sign_classes,
                                                                 h_symmetry_classes=h_symmetry_classes,
                                                                 image_size=(args.input_size, args.input_size),
                                                                 ignore_npz=args.ignore_npz,
                                                                 out_classes=out_classes)

    # TODO: The data are now loaded on numpy array, create a pytorch datset from that


if __name__ == '__main__':
    main()


