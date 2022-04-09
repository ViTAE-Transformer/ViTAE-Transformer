from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .dataset import Dataset
from .cache_image_folder import CachedImageFolder

# def build_dataset(is_train, config):
def build_dataset(directory, idxFilesRoot=None, ZIP_MODE=False):
    # transform = build_transform(is_train, config)
    if idxFilesRoot == '':
        pass
        # ann_file = os.path.join('meta', prefix + "_map.txt")
    else:
        raise NotImplementedError('such dataloader map not implemented yet!')

    if ZIP_MODE:
        prefix = directory.split('/')[-1]
        directory = os.path.join(*directory.split('/')[:-1])
        ann_file = os.path.join('meta', prefix + "_map.txt")
        # if not os.path.exists(ann_file):
        #     raise EnvironmentError(f'no such files: {ann_file}!')
        prefix = prefix + ".zip@/"
        dataset = CachedImageFolder(directory, ann_file, prefix, transform=None,
                        cache_mode='full' if prefix=='train.zip@/' else 'part')
    else:
        dataset = Dataset(directory, idxFilesRoot=idxFilesRoot)


    # if config.DATA.DATASET == 'imagenet':
    #     prefix = 'train' if is_train else 'val'
    #     if config.DATA.ZIP_MODE:
    #         ann_file = prefix + "_map.txt"
    #         prefix = prefix + ".zip@/"
    #         dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
    #                                     cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
    #     else:
    #         # ToDo: test custom_image_folder
    #         root = os.path.join(config.DATA.DATA_PATH, prefix)
    #         dataset = CustomImageFolder(root, transform=transform)
    #     nb_classes = 1000
    # else:
    #     raise NotImplementedError("We only support ImageNet Now.")

    return dataset
