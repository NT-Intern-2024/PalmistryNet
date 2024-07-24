import numpy as np
from keras_preprocessing import image


def random_flip(img, masks, masks2, u=0.5):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
        for i in range(masks.shape[0]):
            masks[i] = image.flip_axis(masks[i], 1)
        for i in range(masks2.shape[0]):
            masks2[i] = image.flip_axis(masks2[i], 1)
    if np.random.random() < u:
        img = image.flip_axis(img, 0)
        for i in range(masks.shape[0]):
            masks[i] = image.flip_axis(masks[i], 0)
        for i in range(masks2.shape[0]):
            masks2[i] = image.flip_axis(masks2[i], 0)
    return img, masks, masks2


def random_rotate(img, masks, masks2, rotate_limit=(-20, 20), u=0.5):
    if np.random.random() < u:
        theta = np.random.uniform(rotate_limit[0], rotate_limit[1])
        img = image.apply_affine_transform(img, theta=theta)
        for i in range(masks.shape[0]):
            masks[i] = image.apply_affine_transform(masks[i], theta=theta)
        for i in range(masks2.shape[0]):
            masks2[i] = image.apply_affine_transform(masks2[i], theta=theta)
    return img, masks, masks2


def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    x = image.apply_affine_transform(x, ty=ty, tx=tx)
    return x


def random_shift(img, masks, masks2, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        for i in range(masks.shape[0]):
            masks[i] = shift(masks[i], wshift, hshift)
        for i in range(masks2.shape[0]):
            masks2[i] = shift(masks2[i], wshift, hshift)
    return img, masks, masks2


def random_zoom(img, masks, masks2, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = image.apply_affine_transform(img, zx=zx, zy=zy)
        for i in range(masks.shape[0]):
            masks[i] = image.apply_affine_transform(masks[i], zx=zx, zy=zy)
        for i in range(masks2.shape[0]):
            masks2[i] = image.apply_affine_transform(masks2[i], zx=zx, zy=zy)
    return img, masks, masks2


def random_shear(img, masks, masks2, intensity_range=(-0.5, 0.5), u=0.5):
    if np.random.random() < u:
        sh = np.random.uniform(-intensity_range[0], intensity_range[1])
        img = image.apply_affine_transform(img, shear=sh)
        for i in range(masks.shape[0]):
            masks[i] = image.apply_affine_transform(masks[i], shear=sh)
        for i in range(masks2.shape[0]):
            masks2[i] = image.apply_affine_transform(masks2[i], shear=sh)
    return img, masks, masks2


def random_gray(img, u=0.5):
    if np.random.random() < u:
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = np.sum(img * coef, axis=2)
        img = np.dstack((gray, gray, gray))
    return img


def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img


def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img


def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return img


def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_augmentation(img, masks, masks2=None):
    img = random_brightness(img, limit=(-0.2, 0.2), u=0.5)
    img = random_contrast(img, limit=(-0.2, 0.2), u=0.5)
    img = random_saturation(img, limit=(-0.2, 0.2), u=0.5)
    img, masks, masks2 = random_rotate(img, masks, masks2, rotate_limit=(-180, 180), u=0.5)
    img, masks, masks2 = random_shear(img, masks, masks2, intensity_range=(-5, 5), u=0.05)
    img, masks, masks2 = random_flip(img, masks, masks2, u=0.5)
    img, masks, masks2 = random_shift(img, masks, masks2, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.05)
    img, masks, masks2 = random_zoom(img, masks, masks2, zoom_range=(0.8, 1.2), u=0.05)
    return img, masks, masks2