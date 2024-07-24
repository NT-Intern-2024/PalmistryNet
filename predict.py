############Test
import argparse
import os,sys
import tensorflow as tf
# from keras.backend import tensorflow_backend
import keras
import keras.backend as K
from utils import define_model, crop_prediction, lowercase
from keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
# TODO: Import pyTorch is essential to make tensorflow see GPUs
import torch
from PIL import Image
from tensorflow.python.keras.engine.functional import Functional
import time
import tracemalloc
import glob
from utils.MyFunction import *
from utils.cubic_final import *



#NOTE : check index
def output_folder(folderpath, index, filename, epochs, img):
    # Check if the image exists
    os.makedirs(folderpath, exist_ok=True)
    # print(mergelayer_folderpath)
    # listimage=[]
    if index == 0:
        cv2.imwrite(f'./{folderpath}/{index}_{filename}_epochs_{epochs}.png',img)
    elif index == 1:
        cv2.imwrite(f'./{folderpath}/{index}_{filename}_epochs_{epochs}.png',img)
        #listimage.append(1)
    elif index == 2:
        cv2.imwrite(f'./{folderpath}/{index}_{filename}_epochs_{epochs}.png',img)
        #listimage.append(1)
    elif index == 3:
        cv2.imwrite(f'./{folderpath}/{index}_{filename}_epochs_{epochs}.png',img)
        #listimage.append(1)
        
    png_files = [file for file in os.listdir(folderpath)]
    #print(f"png_files: {png_files}")
    return png_files
        
    #merge_img(png_files)


def predict(ACTIVATION='ReLU', dropout=0.1, batch_size=32, repeat=4, minimum_kernel=32, 
            epochs=8, iteration=3, crop_size=128, stride_size=3, 
            input_path='', output_path='', DATASET='ALL'):

    pre_show_files = []
    pre_skel_files = []
    pre_colorize_files = []
    pre_merge_files = []
    pre_merge_crop_poly = []
    
    # NOTE: add loop to train model for each layer
    for index in range(0, 4):
        pre_show_files.append([])
        pre_skel_files.append([])
        pre_colorize_files.append([])
        pre_merge_files.append([])
        pre_merge_crop_poly.append([])
        
        exts = ['png', 'jpg', 'tif', 'bmp', 'gif']

        if not input_path.endswith('/'):
            input_path += '/'
        paths = [input_path + i for i in sorted(os.listdir(input_path)) if i.split('.')[-1] in exts]

        # gt_list_out = {}
        # pred_list_out = {}

        #os.makedirs(f"{output_path}/out_layer{index}/", exist_ok=True)

        activation = globals()[ACTIVATION] #Rectified Linear Unit
        
        # model: Functional  = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration) #ทำ Unet ในไฟล์ define_model.py
        model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=activation, iteration=iteration) #get_net function from define_model.py
        
        #model_name = f"testlab-Colorize_Mask_288*288_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
        
        model_name = f"hand_v3_layer{index}_Iteration_{iteration}_cropsize_{crop_size}_epochs_{epochs}"
        
        # NOTE: Specify model

        #model_name = f"Final_Emer_Iteration_3_cropsize_128_epochs_200"
        
        load_path = f"trained_model/{DATASET}/layer{index}/{model_name}.hdf5"
        
        print(f"Load Model: {model_name}")
        print(f"Model Path: {load_path}")
        
        # sys.exit()
        
        model.load_weights(load_path, by_name=False) # Load model from Keras lib
        
        filenames = []
        overlay_bases = []
        for i in tqdm(range(len(paths))):
            filename = '.'.join(paths[i].split('/')[-1].split('.')[:-1])
            overlay_base = f"{input_path}" + f"{filename}.png"
            filenames.append(filename)
            overlay_bases.append(overlay_base)
            
            # os.makedirs(os.path.join('output',filename,'templine_before_sk'), exist_ok=True)
            #print(f"fileneme: {filename}")
            img = Image.open(paths[i])
            image_size = img.size
            img = np.array(img) / 255.
            img = resize(img, [288, 288]) # original dan resize

            patches_pred, new_height, new_width, adjustImg = crop_prediction.get_test_patches(img, crop_size, stride_size)
                        
            # NOTE: Save log (Edit name if needed)
            # print(f"Starting model.predict (Check folder 'logs/custom/' for progress)")
            print(f"Starting model.predict Index: {index}")

            preds = model.predict(patches_pred)  # TODO: <<< FIX OUT OF MEMORY GPU >>>

            print(f"model.predict() - DONE")

            pred = preds[iteration]
            pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:288, 0:288, :]
            probResult = pred_imgs[0, :, :, 0]
            pred_ = probResult
            pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
            pred_seg = pred_
            pred_ = resize(pred_, image_size[::-1])
            # cv2.imwrite(f"{output_path}/out_seg/{filename}.png", pred_)
            
            # print(f"BF art")
            #for artery
            pred = preds[2*iteration + 1]
            pred_patches = crop_prediction.pred_to_patches(pred, crop_size, stride_size)
            pred_imgs = crop_prediction.recompone_overlap(pred_patches, crop_size, stride_size, new_height, new_width)
            pred_imgs = pred_imgs[:, 0:288, 0:288, :]
            probResult = pred_imgs[0, :, :, 0]
            pred_ = probResult
            pred_ = 255. * (pred_ - np.min(pred_)) / (np.max(pred_) - np.min(pred_))
            # pred_art = pred_
            pred_ = resize(pred_, image_size[::-1])
            #cv2.imwrite(f"{output_path}/out_layer{index}/{filename}.png", pred_)
            
            pre_show_files[index].append(pred_)
            
    print(f"filenames = {filenames}")
    # print(f"overlay_base = {overlay_base}")
    
    # for img_num in range(len(filenames)):
    #     warp_image(f"./data/test_preprocess/{filenames[img_num]}.png", f"./data/test_out/{filenames[img_num]}_warped.png")
    #     crop_fingers(f"./data/test_out/IMG_FEMALE_000{idx}_warped.png", f"./data/test_out/IMG_FEMALE_000{idx}_croped.png")
    
    final_files=[[] for n in range(len(filenames))]
    final_files_poly=[[] for n in range(len(filenames))]
    final_files_crop_poly=[[] for n in range(len(filenames))]
    
    for idx in range(0,4):
        for img_num in range(len(filenames)):
            post_show_file = pre_show_files[idx][img_num]
            pre_skel_files[idx] = output_folder(folderpath = f'./output/{filenames[img_num]}/step0_line_show/', index = idx, filename = filenames[img_num], epochs = epochs, img = post_show_file)

        for img_num in range(len(filenames)):
            image_path = f'./output/{filenames[img_num]}/step0_line_show/{idx}_{filenames[img_num]}_epochs_{epochs}.png'
            post_skel_file = convert_to_skel(image_path = image_path)
            pre_colorize_files[idx] = output_folder(folderpath = f'./output/{filenames[img_num]}/step1_line_skel/', index = idx, filename = filenames[img_num], epochs = epochs, img = post_skel_file)
            
        for img_num in range(len(filenames)):
            image_path = f'./output/{filenames[img_num]}/step1_line_skel/{idx}_{filenames[img_num]}_epochs_{epochs}.png'
            post_colorize_file = colorize_lines(image_path = image_path, index = idx)
            pre_merge_files[idx] = output_folder(folderpath = f'./output/{filenames[img_num]}/step2_line_color/', index = idx, filename = filenames[img_num], epochs = epochs, img = post_colorize_file)
            
            image_path_for_color = f'./output/{filenames[img_num]}/step2_line_color/{idx}_{filenames[img_num]}_epochs_{epochs}.png'
            final_files[img_num].append(image_path_for_color)

    print(f"Saving Coefficients and Fitted Cubic Equation at './output/coefficients.txt'")
    close_stdout = sys.stdout
    sys.stdout = open(f'./output/coefficients.txt', 'w')
    
    for img_num in range(len(filenames)):
        post_merge_file = merge_images(final_files[img_num])
        os.makedirs(f'./output/{filenames[img_num]}/step3_line_final/', exist_ok=True)
        cv2.imwrite(f'./output/{filenames[img_num]}/step3_line_final/{filenames[img_num]}_epochs_{epochs}.png', post_merge_file)
        os.makedirs(f'./output/{filenames[img_num]}/step4_poly/', exist_ok=True)
        perform_cubic_regression(final_files[img_num], f'./output/{filenames[img_num]}/step4_poly/', save_coef=True)
        print(f" ")
        
        for idx in range(0,4):
            image_path_for_poly = f'./output/{filenames[img_num]}/step4_poly/{idx}_{filenames[img_num]}_epochs_{epochs}.png'
            final_files_poly[img_num].append(image_path_for_poly)
            
        post_merge_poly = merge_images4poly(final_files_poly[img_num], overlay_bases[img_num])
        os.makedirs(f'./output/{filenames[img_num]}/step5_merge_poly/', exist_ok=True)
        cv2.imwrite(f'./output/{filenames[img_num]}/step5_merge_poly/{filenames[img_num]}_epochs_{epochs}.png', post_merge_poly)

        os.makedirs(f'./output/{filenames[img_num]}/step6_highlight/', exist_ok=True)
        perform_cubic_regression(final_files[img_num], f'./output/{filenames[img_num]}/step6_highlight/', color_thick=10)
        
        for idx in range(0,4):
            image_path = f'./output/{filenames[img_num]}/step6_highlight/{idx}_{filenames[img_num]}_epochs_{epochs}.png'
            post_crop_excess = crop_excess_line(image_path = image_path, idx = idx)
            pre_merge_crop_poly[idx] = output_folder(folderpath = f'./output/{filenames[img_num]}/step7_crop_excess/', index = idx, filename = filenames[img_num], epochs = epochs, img = post_crop_excess)
        
            image_path_for_crop_poly = f'./output/{filenames[img_num]}/step7_crop_excess/{idx}_{filenames[img_num]}_epochs_{epochs}.png'
            final_files_crop_poly[img_num].append(image_path_for_crop_poly)

    sys.stdout.close()
    sys.stdout = close_stdout

    for img_num in range(len(filenames)):
        post_merge_crop_poly = merge_images4poly(final_files_crop_poly[img_num], overlay_bases[img_num])
        os.makedirs(f'./output/{filenames[img_num]}/step8_merge_crop_poly/', exist_ok=True)
        cv2.imwrite(f'./output/{filenames[img_num]}/step8_merge_crop_poly/{filenames[img_num]}_epochs_{epochs}.png', post_merge_crop_poly)
        
            # if isinstance(png_names, list):
            #     if len(png_names) >=4:
            #         line_before_skel(png_names,filename,epochs)
            # else:
            #     print(f"{filename} not layer 3 yet")

if __name__ == "__main__":

    print('Enter main function')

    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print(f"No GPU detected.")
    else:
        print(f"GPU detected.")
        try:
            for gpu in gpus:
                print(gpu)
                tf.config.experimental.set_memory_growth(gpu, True)
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True

            sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(sess)

        except RuntimeError as e:
            print(e)

    des_text = 'Please use -i to specify the input dir and -o to specify the output dir.'
    
    parser = argparse.ArgumentParser(description=des_text)
    parser.add_argument('--input', '-i', help="(Required) Path of input dir", required=True)
    parser.add_argument('--output', '-o', help="(Optional) Path of output dir")
    # TODO: parse_args
    args = parser.parse_args(["--input", "./data/test_images/", "--output", "./output/"])

    if not args.input:
        print('Please specify the input dir with -i')
        exit(1)

    input_path = args.input

    if not args.output:
        output_path = './output/'
    else:
        output_path = args.output
        if output_path.endswith('/'):
            output_path = output_path[:-1]

    batch_size=32
    epochs=100
    iteration=3
    stride_size=3
    crop_size=128

    predict(batch_size=batch_size, epochs=epochs, iteration=iteration, stride_size=stride_size,
            crop_size=crop_size, input_path=input_path, output_path=output_path)