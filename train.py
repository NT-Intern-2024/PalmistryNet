import numpy as np
import os, sys
import time
from datetime import datetime
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import ReLU
from keras.utils import plot_model
from utils import define_model, MyFunction, prepare_dataset
import tensorflow as tf

def train(iteration=3, DATASET='ALL', crop_size=128, need_au=True, ACTIVATION='ReLU', dropout=0.1, batch_size=32,
          repeat=4, minimum_kernel=32, epochs=20):
    
    # NOTE: add loop to predict each layer
    for index in range(1, 2):
        # model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
    
        # NOTE: Specify model
        model_name = f"hand_v3_layer{index}_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"

        prepare_dataset.prepareDataset(DATASET, index=index)
        
        activation = globals()[ACTIVATION]
        model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration)

        try:
            os.makedirs(f"trained_model/{DATASET}/layer{index}/", exist_ok=True)
            os.makedirs(f"logs/{DATASET}/layer{index}/", exist_ok=True)
        except:
            pass

        # NOT using
        # load_path = f"trained_model/{DATASET}/{model_name}_weights.best.hdf5"
        # try:
        #     model.load_weights(load_path, by_name=True)
        # except:
        #     pass
        
        now = datetime.now() # current date and time
        date_time = now.strftime("%Y-%m-%d---%H-%M-%S")
        
        tensorboard = TensorBoard(
            log_dir=f"logs/{DATASET}/hand_v1_layer{index}_Iteration_{iteration}-Cropsize_{crop_size}-Epochs_{epochs}---{date_time}",
            histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_metadata=None, update_freq='epoch')

        save_path = f"trained_model/{DATASET}/layer{index}/{model_name}.hdf5"
        print("Model : %s" % model_name)
        
        checkpoint = ModelCheckpoint(save_path, monitor='seg_final_out_loss', verbose=1, save_best_only=True, mode='min')

        data_generator = define_model.Generator(batch_size, repeat, DATASET, index=index)

        # NOTE: Save log (Edit name if needed)
        # print(f"Starting model.fit (Check folder 'logs/custom/' for progress)")
        
        print(f"Starting model.fit Index: {index}")
        
        #close_stdout = sys.stdout
        #sys.stdout = open('./logs/custom/non-colorize_mask_288*288_train_log.txt', 'w')
        #tf.profiler.experimental.start('./logdir')
        #sys.exit()
        
        history = model.fit(data_generator.gen(au=need_au, crop_size=crop_size, iteration=iteration),
                                    epochs=epochs, verbose=1,
                                    steps_per_epoch=100 * data_generator.n // batch_size,
                                    callbacks=[tensorboard, checkpoint])
        
        print(f"model.fit() - DONE")
        
        #tf.profiler.experimental.stop()
        #sys.stdout.close()
        #sys.stdout = close_stdout

if __name__ == "__main__":
    # This part of the code was modified due to the tensorflow version 2.15.
    gpus = tf.config.list_physical_devices('GPU')
    print("gpu devices: ", gpus)
    
    if not gpus:
        print(f"No GPU detected.")
    else:
        print(f"GPU detected.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)

    batch_size=32
    iteration=3
    epochs=100
    crop_size=128
    train(batch_size=batch_size, iteration=iteration, epochs=epochs, crop_size=crop_size)