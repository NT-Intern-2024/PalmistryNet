import numpy as np
import sys

def get_test_patches(img, crop_size, stride_size, rl=False):
    # test_img = []
    # test_img.append(img)
    # test_img = np.asarray(test_img)
    
    # NOTE: Refined the function by adding support for GRAYSCALE images
    if len(img.shape) == 2: # If the image is GRAYSCALE
        print(f"INPUT IMAGES IS GRAYSCALES")
        img = np.expand_dims(img, axis=-1) # Add the channels dimension
    # Change color channel from 1 to 3
    if img.shape[-1] == 1: 
        img = np.repeat(img, 3, axis=-1)
    
    test_img = np.expand_dims(img, axis=0)
    
    # print("load image to array func get_test_patches()")
    
    # test_img_adjust=img_process(test_img,rl=rl)
    test_img_adjust = test_img
    test_imgs = paint_border(test_img_adjust, crop_size, stride_size)

    test_img_patch = extract_patches(test_imgs, crop_size, stride_size)

    return test_img_patch, test_imgs.shape[1] , test_imgs.shape[2] , test_img_adjust


def extract_patches(full_imgs, crop_size, stride_size):
    # print(f"Ent extract_patches func")

    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size

    assert (len(full_imgs.shape) == 4)  # 4D arrays
    img_h = full_imgs.shape[1] # height of the full image
    img_w = full_imgs.shape[2] # width of the full image

    assert ((img_h - patch_height) % stride_height == 0 and (img_w - patch_width) % stride_width == 0)
    N_patches_img = ((img_h - patch_height) // stride_height + 1) * (
            (img_w - patch_width) // stride_width + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]

    patches = np.empty((N_patches_tot, patch_height, patch_width, full_imgs.shape[3]))
    # patches = np.zeros((N_patches_tot, patch_height, patch_width, full_imgs.shape[3]))
    
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_height) // stride_height + 1):
            for w in range((img_w - patch_width) // stride_width + 1):
                patch = full_imgs[i, h * stride_height:(h * stride_height) + patch_height,
                        w * stride_width:(w * stride_width) + patch_width, :]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    
    print(f"extracted_patches value size (64 bit): {sys.getsizeof(patches)}")
    # print("extracted_patches return value size 32 bit",sys.getsizeof(patches.astype(np.float32)))
    return patches #this retuen value use in model.predict()  lib keras


def paint_border(imgs, crop_size, stride_size):
    # print(f"Ent paint border func")
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size

    assert (len(imgs.shape) == 4)
    # img_h = imgs.shape[1] # height of the full image
    # img_w = imgs.shape[2] # width of the full image
    img_h = imgs.shape[1] # height of the full image
    img_w = imgs.shape[2] # width of the full image
    leftover_h = (img_h - patch_height) % stride_height  # leftover on the h dim
    leftover_w = (img_w - patch_width) % stride_width  # leftover on the w dim
    full_imgs = None
    if (leftover_h != 0):  # change dimension of img_h
        tmp_imgs = np.zeros((imgs.shape[0], img_h + (stride_height - leftover_h), img_w, imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:img_h, 0:img_w, 0:imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
    if (leftover_w != 0):  # change dimension of img_w
        tmp_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], img_w + (stride_width - leftover_w), full_imgs.shape[3]))
        tmp_imgs[0:imgs.shape[0], 0:imgs.shape[1], 0:img_w, 0:full_imgs.shape[3]] = imgs
        full_imgs = tmp_imgs
        #     print("new full images shape: \n" +str(full_imgs.shape))
        return full_imgs
    else:
        return imgs


def pred_to_patches(pred, crop_size, stride_size):
    return pred
    patch_height = crop_size
    patch_width = crop_size

    seg_num = 0
    #     print(pred.shape)

    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)

    pred_images = np.empty((pred.shape[0], pred.shape[1], seg_num + 1))  # (Npatches,height*width)
    pred_images[:, :, 0:seg_num + 1] = pred[:, :, 0:seg_num + 1]
    pred_images = np.reshape(pred_images, (pred_images.shape[0], patch_height, patch_width, seg_num + 1))
    return pred_images


def recompone_overlap(preds, crop_size, stride_size, img_h, img_w):
    assert (len(preds.shape) == 4)  # 4D arrays

    patch_h = crop_size
    patch_w = crop_size
    stride_height = stride_size
    stride_width = stride_size

    N_patches_h = (img_h - patch_h) // stride_height + 1
    N_patches_w = (img_w - patch_w) // stride_width + 1
    N_patches_img = N_patches_h * N_patches_w
    #     print("N_patches_h: " +str(N_patches_h))
    #     print("N_patches_w: " +str(N_patches_w))
    #     print("N_patches_img: " +str(N_patches_img))
    # assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0] // N_patches_img
    #     print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = np.zeros(
        (N_full_imgs, img_h, img_w, preds.shape[3]))  # itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, img_h, img_w, preds.shape[3]))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_height + 1):
            for w in range((img_w - patch_w) // stride_width + 1):
                full_prob[i, h * stride_height:(h * stride_height) + patch_h,
                w * stride_width:(w * stride_width) + patch_w, :] += preds[k]
                full_sum[i, h * stride_height:(h * stride_height) + patch_h,
                w * stride_width:(w * stride_width) + patch_w, :] += 1
                k += 1
    #     print(k,preds.shape[0])
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    #     print('using avg')
    return final_avg