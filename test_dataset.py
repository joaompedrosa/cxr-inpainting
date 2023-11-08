import copy
import os
import random
from matplotlib import pyplot as plt
from shutil import copyfile
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils

import data.dataset as dataset
dataset.storeOnRAM = False
from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list

def get_configs(args):
    # Get configs
    config_test = get_config(args['config'])
    config_train = get_config(os.path.join(args["checkpoint_path"], 'config.yaml'))

    # CUDA configuration
    device_ids = config_test['gpu_ids']
    if config_test['cuda']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config_test['gpu_ids'] = device_ids
        cudnn.benchmark = True
    print("Arguments: {}".format(args))

    # Set random seed
    if args['seed'] is None:
        args['seed'] = random.randint(1, 10000)
    print("Random seed: {}".format(args['seed']))
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if config_test['cuda']:
        torch.cuda.manual_seed_all(args['seed'])

    if 'ref_func' in config_train:
        if config_train['ref_func'] is None:
            config_test['ref_func'] = config_train['ref_func']
        else:
            reffunc = config_train['ref_func'].split('_')
            reffunc[0] = args['dataset']
            config_test['ref_func'] = '_'.join(reffunc)
        print(f'Preparing inference with reference function {config_test["ref_func"]}')

    if 'addmask_suff' in config_train:
        config_test['addmask_suff'] = config_train['addmask_suff']

    print("Configuration: {}".format(config_test))

    return config_test, config_train

def make_dataloader(args, config_test):
    datapath = os.path.join('media', args['dataset'])
    if 'ref_func' in config_test:
        if 'mask_csv' in config_test:
            test_dataset = dataset.Dataset_BboxCsv_RefCXR(data_path=datapath,
                                                  with_subfolder=config_test['data_with_subfolder'],
                                                  image_shape=config_test['image_shape'],
                                                  bbox_csvfname=config_test['mask_csv'],
                                                  random_crop=config_test['random_crop'],
                                                  return_name=True,
                                                  min_mask_shape=config_test["min_mask_shape"],
                                                  ref_func=config_test['ref_func'])
        else:
            test_dataset = dataset.Dataset_RefCXR(data_path=datapath,
                                          with_subfolder=config_test['data_with_subfolder'],
                                          image_shape=config_test['image_shape'],
                                          random_crop=config_test['random_crop'],
                                          return_name=True,
                                          ref_func=config_test['ref_func'])
    else:
        if 'mask_csv' in config_test:
            if 'addmask_suff' in config_test:
                test_dataset = dataset.Dataset_BboxCsv_MaskCXR(data_path=datapath,
                                                      with_subfolder=config_test['data_with_subfolder'],
                                                      image_shape=config_test['image_shape'],
                                                      bbox_csvfname=config_test['mask_csv'],
                                                      random_crop=config_test['random_crop'],
                                                      return_name=True,
                                                      min_mask_shape=config_test["min_mask_shape"],
                                                      addmask_suff=config_test['addmask_suff'])
            else:
                test_dataset = dataset.Dataset_BboxCsv(data_path=datapath,
                                               with_subfolder=config_test['data_with_subfolder'],
                                               image_shape=config_test['image_shape'],
                                               bbox_csvfname=config_test['mask_csv'],
                                               random_crop=config_test['random_crop'],
                                               return_name=True,
                                               min_mask_shape=config_test["min_mask_shape"])
        else:
            if 'addmask_suff' in config_test:
                test_dataset = dataset.Dataset_MaskCXR(data_path=datapath,
                                               with_subfolder=config_test['data_with_subfolder'],
                                               image_shape=config_test['image_shape'],
                                               random_crop=config_test['random_crop'],
                                               return_name=True,
                                               addmask_suff=config_test['addmask_suff'])
            else:
                test_dataset = dataset.Dataset(data_path=datapath,
                                       with_subfolder=config_test['data_with_subfolder'],
                                       image_shape=config_test['image_shape'],
                                       random_crop=config_test['random_crop'],
                                       return_name=True)

    print(f'Preparing predictions on {len(test_dataset)} images.')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=10,
                                              shuffle=False,
                                              num_workers=config_test['num_workers'])

    return test_loader

def get_generator(config_train, config_test):
    # Define the trainer
    netG = Generator(config_train['netG'], config_test['cuda'], config_test['gpu_ids'])
    if config_test['cuda']:
        netG = nn.parallel.DataParallel(netG, device_ids=config_test['gpu_ids'])

    return netG

def get_model_names(args, config_test):
    print(f'Loading model from {args["checkpoint_path"]}')
    last_model_names = get_model_list(args["checkpoint_path"], "gen", iteration=args['iter'])
    if isinstance(last_model_names, str):
        last_model_names = [last_model_names]

    if not args['disp']:
        last_model_names = [lmname for lmname in last_model_names
                            if not os.path.isdir(os.path.join(args["checkpoint_path"], f'{args["dataset"]}_{int(lmname[-11:-3])}_v{config_test["test_ver"]}'))]

    print(f'Preparing inference for iterations: {[int(lmname[-11:-3]) for lmname in last_model_names]}')

    return last_model_names

def run_inference(config_test, config_train, netG, batch):
    if 'mask_csv' in config_test:
        if 'ref_func' in config_test or 'addmask_suff' in config_test:
            fnames, ground_truth, reference, bboxes, bboxes_cl = batch
        else:
            fnames, ground_truth, bboxes, bboxes_cl = batch
            reference = None
        bboxes = bboxes.to(torch.int)
    else:
        if 'ref_func' in config_test or 'addmask_suff' in config_test:
            fnames, ground_truth, reference = batch
        else:
            fnames, ground_truth = batch
            reference = None
        bboxes = random_bbox(config_test, batch_size=ground_truth.size(0))
        bboxes_cl = None

    if 'force_min_mask_shape' in config_test and config_test['force_min_mask_shape']:
        for bb in bboxes:
            if bb[2] < config_test['min_mask_shape'][0]:
                bbc = int(bb[0] + bb[2]/2)
                bb[0] = bbc - int(config_test['min_mask_shape'][0]/2)
                bb[2] = config_test['min_mask_shape'][0]
            if bb[3] < config_test['min_mask_shape'][1]:
                bbc = int(bb[1] + bb[3]/2)
                bb[1] = bbc - int(config_test['min_mask_shape'][1]/2)
                bb[3] = config_test['min_mask_shape'][1]

    x, mask = mask_image(ground_truth, bboxes, config_test)

    if 'lesion_data_path' in config_train and not config_train['lesion_data_path'] is None:
        x = copy.deepcopy(ground_truth)

    if config_test['cuda']:
        x = x.cuda()
        mask = mask.cuda()
        if 'ref_func' in config_test:
            reference = reference.cuda()


    # Inference
    x1, x2, _ = netG(x, mask, reference)

    return fnames, ground_truth, bboxes, bboxes_cl, x, mask, x1, x2

def main(args):
    config_test, config_train = get_configs(args)
    try:
        last_model_names = get_model_names(args, config_test)
    except:
        return

    if not last_model_names:
        return

    test_loader = make_dataloader(args, config_test)
    netG = get_generator(config_train, config_test)

    for last_model_name in last_model_names:
        # Resume weight
        netG.load_state_dict(torch.load(last_model_name, map_location='cuda'))
        model_iteration = int(last_model_name[-11:-3])
        print(f'Running config {config_test["test_ver"]} for iteration {model_iteration}')

        # Create save folder
        if not args['disp']:
            foldername = f'{args["dataset"]}_{model_iteration}_v{config_test["test_ver"]}'
            os.mkdir(os.path.join(args["checkpoint_path"], foldername))
            nppath = os.path.join(args["checkpoint_path"], foldername, 'npy')
            os.mkdir(nppath)
            copyfile(args['config'], os.path.join(args["checkpoint_path"], foldername, 'config_test.yaml'))

        for b_ind, batch in enumerate(test_loader):
            print(f'Batch {b_ind}/{len(test_loader)}...')
            for r_ind in range(args['nreps_batch']):
                fnames, ground_truth, bboxes, bboxes_cl, x, mask, x1, x2 = run_inference(config_test, config_train, netG, batch)

                if args['disp']:
                    import matplotlib
                    matplotlib.use('TkAgg')
                    for i in range(x.shape[0]):
                        plt.figure()
                        plt.imshow(x[i, 0, :, :].detach().numpy(), 'gray')
                        plt.figure()
                        plt.imshow(mask[i, 0, :, :].detach().numpy(), 'gray')
                        plt.figure()
                        plt.imshow(x1[i, 0, :, :].detach().numpy(), 'gray')
                        plt.figure()
                        plt.imshow(x2[i, 0, :, :].detach().numpy(), 'gray')
                        plt.show()
                else:
                    np.save(os.path.join(nppath, f'batch_{b_ind}_{r_ind}_pd.npy'), x2.detach().numpy())
                    np.save(os.path.join(nppath, f'batch_{b_ind}_{r_ind}_bboxes.npy'), np.array(bboxes))
                    if not bboxes_cl is None:
                        np.save(os.path.join(nppath, f'batch_{b_ind}_0_bboxescl.npy'), np.array(bboxes_cl))
                    if r_ind == 0:
                        np.save(os.path.join(nppath, f'batch_{b_ind}_{r_ind}_gt.npy'), ground_truth.detach().numpy())
                        np.save(os.path.join(nppath, f'batch_{b_ind}_{r_ind}_fnames.npy'), np.array(fnames)[:, None])
                        if b_ind == 0:
                            viz_images = torch.stack([x2[:16], mask[:16]], dim=1)
                            viz_images = viz_images.view(-1, *list(x.size())[1:])
                            vutils.save_image(viz_images, f'{os.path.join(args["checkpoint_path"], foldername)}/batch0.png',
                                              nrow=2 * 4, normalize=True)


if __name__ == '__main__':
    args = {}
    args['config'] = 'configs/test.yaml'
    args['dataset'] = 'test'
    args['seed'] = None
    args['disp'] = True # if True plot only, if False saves results to checkpoint_path
    args['nreps_batch'] = 1
    args["checkpoint_path"] = f'checkpoints\\anacattnet-ar'
    args['iter'] = 500000
    main(args)