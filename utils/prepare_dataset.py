import h5py
import numpy as np
import os.path
from PIL import Image
from glob import glob
from skimage.transform import resize
import sys

# NOTE: add layers
raw_data_path_all = []
HDF5_data_path_all = []
for i in range(0, 4):
    raw_data_path_layer = []
    raw_training_x_path = f'./data/ALL/layer{i}/training/images/*.png'
    raw_training_y_path = f'./data/ALL/layer{i}/training/av/*.png'
    raw_data_path_layer = [raw_training_x_path, raw_training_y_path] # From original
    raw_data_path_all.append(raw_data_path_layer)
    # HDF5_data_path_layer = f'./data/HDF5/layer{i}/'
    # HDF5_data_path_all.append(HDF5_data_path_layer)

HDF5_data_path = f'./data/HDF5/'

DESIRED_DATA_SHAPE = [288, 288]

# NOTE: second call in train.py
def isHDF5exists(raw_data_path_all, HDF5_data_path):
    print(f'raw_data_path{raw_data_path_all}')
    #sys.exit()
    for raw in raw_data_path_all:
        if not raw:
            continue

        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/*.hdf5'])

        if len(glob(HDF5)) == 0:
            return False

    return True


def read_input(path):
    x = np.array(Image.open(path)) / 255.
    if x.shape[-1] == 3:
        return x
    else:
        return x[..., np.newaxis]


def read_input_av_label(path, need_one_hot=False):
    global DESIRED_DATA_SHAPE
    x = Image.open(path)
    x = np.array(x) / 255.
    x = resize(x, DESIRED_DATA_SHAPE)

    new_x_whole = np.zeros((x.shape[0], x.shape[1]))
    new_x_a = np.zeros((x.shape[0], x.shape[1]))
    new_x_v = np.zeros((x.shape[0], x.shape[1]))
    new_x_cross= np.zeros((x.shape[0], x.shape[1]))
    new_x_unknown = np.zeros((x.shape[0], x.shape[1]))
    new_x_onehot = np.zeros((x.shape[0], x.shape[1], 3)) #bg, a, v

    for row_id, row in enumerate(x):
        for col_id, elem in enumerate(row):
            if elem[0] > 0.5 or elem[1] > 0.5 or elem[2] > 0.5:
                new_x_whole[row_id, col_id] = 1.0
            else:
                new_x_onehot[row_id, col_id] = np.array((1.0, 0.0, 0.0))
            if elem[0] > 0.5 and elem[1] > 0.5 and elem[2] > 0.5:
                new_x_unknown[row_id, col_id] = 1.0
                new_x_a[row_id, col_id] = 1.0
                new_x_onehot[row_id, col_id] = np.array((0.0, 1.0, 0.0))
            elif elem[0] > 0.5:
                new_x_a[row_id, col_id] = 1.0
                new_x_onehot[row_id, col_id] = np.array((0.0, 1.0, 0.0))
            elif elem[1] > 0.5:
                new_x_cross[row_id, col_id] = 1.0
                new_x_a[row_id, col_id] = 1.0
                # new_x_v[row_id, col_id] = 1.0
                new_x_onehot[row_id, col_id] = np.array((0.0, 1.0, 0.0))
            elif elem[2] > 0.5:
                new_x_v[row_id, col_id] = 1.0
                new_x_onehot[row_id, col_id] = np.array((0.0, 0.0, 1.0))
    
    new_x_whole = resize(new_x_whole[..., np.newaxis], DESIRED_DATA_SHAPE)
    new_x_a = resize(new_x_a[..., np.newaxis], DESIRED_DATA_SHAPE)
    new_x_v = resize(new_x_v[..., np.newaxis], DESIRED_DATA_SHAPE)
    new_x_cross = resize(new_x_cross[..., np.newaxis], DESIRED_DATA_SHAPE)
    new_x_unknown = resize(new_x_unknown[..., np.newaxis], DESIRED_DATA_SHAPE)
    new_x_onehot = resize(new_x_onehot, DESIRED_DATA_SHAPE)

    if not need_one_hot:
        # print([new_x_whole, new_x_cross, new_x_a, new_x_v,  new_x_unknown])
        return [new_x_whole, new_x_cross, new_x_a, new_x_v,  new_x_unknown]
    else:
        # print([new_x_onehot])
        return [new_x_onehot]
    

# NOTE: third call function
def preprocessData(data_path, dataset, need_one_hot=False):
    print('ENTER preprocessData')
    
    global DESIRED_DATA_SHAPE

    print(f"data_path = {data_path}")
    data_path = list(sorted(glob(data_path)))
    print(f"data_path = {data_path}")

    #print('this is datapath0',data_path[0])
    
    # NOTE :ORIGINAL CODE
    # if data_path[0].find('/mask/') > 0:
    #     return np.array([read_input(image_path) for image_path in data_path]) # From original
    # elif data_path[0].find('/av/') > 0 or data_path[0].find('/arteries-and-veins/') > 0 :    
    #     print(f"av")
    #     return np.array([read_input_av_label(image_path, need_one_hot) for image_path in data_path])
    # else:
    #     print(f"other")
    #     return np.array([resize(read_input(image_path), DESIRED_DATA_SHAPE) for image_path in data_path])
    
    # NOTE: LEE not use /mask/ 
    if data_path[0].find('/av/') > 0 or data_path[0].find('/arteries-and-veins/') > 0 :    
        print(f"av")
        #sys.exit()
        return np.array([read_input_av_label(image_path, need_one_hot) for image_path in data_path])
    else:
        print(f"other")
        return np.array([resize(read_input(image_path), DESIRED_DATA_SHAPE) for image_path in data_path])

# NOTE : call fourth  in train.py
def createHDF5(data, HDF5_data_path, one_hot=False):
    try:
        os.makedirs(HDF5_data_path, exist_ok=True)
    except:
        pass
    if not one_hot:
        f = h5py.File(HDF5_data_path + 'data.hdf5', 'w')
        print('create hdf5 images')
    else:
        f = h5py.File(HDF5_data_path + 'data_onehot.hdf5', 'w')
        print('create hdf5')
    f.create_dataset('data', data=data)
    
    
    return

# NOTE: first call in train.py
def prepareDataset(dataset, index):
    print('ENTER prepare dataset')
    
    global raw_data_path_all, HDF5_data_path
    global DESIRED_DATA_SHAPE
  
    print(f"raw_data_path: {raw_data_path_all[index]}")
    print(f"HDF5_data_path: {HDF5_data_path}")
    
    # NOTE: Comment to avoid case that might result in error
    # if isHDF5exists(raw_data_path_all[index], HDF5_data_path_all[index]):
    #     print(f"End of prepareDataset func")
    #     return

    for raw in raw_data_path_all[index]:
        if not raw:
            continue

        raw_splited = raw.split('/')
        HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/'])

        preprocessed = preprocessData(raw, dataset)
        createHDF5(preprocessed, HDF5)

        if raw.find('/av/') > 0 or raw.find('/arteries-and-veins/') > 0:
            raw_splited = raw.split('/')
            HDF5 = ''.join([HDF5_data_path, '/'.join(raw_splited[2:-1]), '/'])

            preprocessed = preprocessData(raw, dataset, need_one_hot=True)
            createHDF5(preprocessed, HDF5, one_hot=True)

# NOTE: use in class Generator in file define model.py
def getTrainingData(XorY, dataset, index, need_one_hot=False):
    global HDF5_data_path
    
    raw_training_x_path, raw_training_y_path = raw_data_path_all[index][:2]

    if XorY == 0:
        raw_splited = raw_training_x_path.split('/')
    else:
        raw_splited = raw_training_y_path.split('/')

    if not need_one_hot:
        data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
    else:
        data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data_onehot.hdf5'])
    f = h5py.File(data_path, 'r')
    data = f['data']

    return data

# NOTE: Not using but existed in original file
# def getTestData(XorYorMask, dataset, need_one_hot=False):
#     global HDF5_data_path
    
#     raw_test_x_path, raw_test_y_path = raw_data_path[2:]

#     if XorYorMask == 0:
#         raw_splited = raw_test_x_path.split('/')
#     elif XorYorMask == 1:
#         raw_splited = raw_test_y_path.split('/')
#     else:
#         if not raw_test_mask_path:
#             return None
#         raw_splited = raw_test_mask_path.split('/')

#     if not need_one_hot:
#         data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data.hdf5'])
#     else:
#         data_path = ''.join([HDF5_data_path, dataset, '/', '/'.join(raw_splited[3:-1]), '/data_onehot.hdf5'])
#     f = h5py.File(data_path, 'r')
#     data = f['data']

#     return data