# PyTorch
from torch.utils.data import DataLoader
from torch import no_grad
from torch import load as torchload

import os
import time
import numpy as np
from matplotlib import pyplot as plt

import segmentation.utils_data
from segmentation.utils_model import get_model, save_model_params, load_model_params
from segmentation.utils_transforms import TestTransform
from segmentation.utils_train import dataPredictions


def main(args):
    # Load model params
    model_params = load_model_params(args['model'])

    # Image transformations
    image_transform = TestTransform(model_params)

    dataset, path_results = loadData(args['dataset'], image_transform)

    model = loadModel(args['model'], model_params)
    dataloader = getDataLoader(dataset)
    predict(model, dataloader, path_results, dispFlag=args['disp'])


def loadData(datastr, image_transform):
    # Path to results
    path_results = os.path.join('media', datastr + '_antribsegm')
    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    # Datasets from each folder
    dataset = segmentation.utils_data.DatasetFolder(root=datastr, transform=image_transform)
    print('Found {} samples.'.format(len(dataset.samples)))

    return dataset, path_results


def loadModel(model_folder, model_params):
    model_path = os.path.join(model_folder, 'model_0.pt')
    print('Loading', model_path)
    # Load model
    model = get_model(model_params)
    model_dict = torchload(model_path)
    model_dict = {k: model_dict[k] for k in model.state_dict()}
    model.load_state_dict(model_dict)

    return model


def getDataLoader(dataset):
    # Dataloader iterators
    dataloader = DataLoader(dataset, batch_size=10)

    return dataloader


def predict(model, dataloader, path_results, dispFlag=False):
    dPredictions = dataPredictions()
    with no_grad():
        model.eval()
        nbatches = len(dataloader)
        st_time = time.time()
        for ii, (data, target, info) in enumerate(dataloader):

            output = model(data)

            if dispFlag:
                for d, o in zip(data, output):
                    image = d.cpu().permute(1, 2, 0).detach().numpy()
                    segm = o.cpu().permute(1, 2, 0).detach().numpy()
                    fig, (ax0, ax1) = plt.subplots(1, 2)
                    ax0.imshow(image)
                    ax1.imshow(segm)
                    plt.show()

            dPredictions.append(data, output, target=target, info=info)
            print(f'\rSegmentations: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds '
                  f'elapsed in fold.', end='')
            dPredictions.write(path_results, clear=True)

    print(f'Segmentations: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed '
          f'in fold.')


if __name__ == '__main__':
    args = {'model': 'segmentation\\model',
            'dataset': 'test',
            'disp': False}
    main(args)
