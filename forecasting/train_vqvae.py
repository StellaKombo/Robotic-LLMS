import argparse
import comet_ml
import json
import numpy as np
import os
import pdb
import random
import time
import torch
from lib.models import get_model_class
from time import gmtime, strftime


def main(device, config, save_dir, logger, data_init_loc, args):
    if os.path.exists(os.path.join(save_dir, 'checkpoints')):
        print('Checkpoint Directory Already Exists - skipping overwrite prompt.')
    else:
        os.makedirs(os.path.join(save_dir, 'checkpoints'))


    if logger is not None:
        logger.log_parameters(config)

    vqvae_config, summary = start_training(device=device, vqvae_config=config['vqvae_config'], save_dir=save_dir,
                                           logger=logger, data_init_loc=data_init_loc, args=args)

    config['vqvae_config'] = vqvae_config
    print('CONFIG FILE TO SAVE:', config)

    if os.path.exists(os.path.join(save_dir, 'configs')):
        print('Saved Config Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(save_dir, 'configs'))

    with open(os.path.join(save_dir, 'configs', 'config_file.json'), 'w+') as f:
        json.dump(config, f, indent=4)

    summary['log_path'] = os.path.join(save_dir)
    master['summaries'] = summary
    print('MASTER FILE:', master)
    with open(os.path.join(save_dir, 'master.json'), 'w') as f:
        json.dump(master, f, indent=4)


def start_training(device, vqvae_config, save_dir, logger, data_init_loc, args):
    summary = {}
    general_seed = args.seed
    summary['general_seed'] = general_seed
    torch.manual_seed(general_seed)
    random.seed(general_seed)
    np.random.seed(general_seed)

    torch.backends.cudnn.deterministic = True

    summary['data initialization location'] = data_init_loc
    summary['device'] = device

    model_class = get_model_class(vqvae_config['model_name'].lower())
    model = model_class(vqvae_config)

    print('Total # trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if vqvae_config['pretrained']:
        model = torch.load(vqvae_config['pretrained'])
    summary['vqvae_config'] = vqvae_config

    start_time = time.time()
    model = train_model(model, device, vqvae_config, save_dir, logger, args=args)

    torch.save(model, os.path.join(save_dir, 'checkpoints/final_model.pth'))
    summary['total_time'] = round(time.time() - start_time, 3)
    return vqvae_config, summary


def train_model(model, device, vqvae_config, save_dir, logger, args):
    optimizer = model.configure_optimizers(lr=vqvae_config['learning_rate'])
    model.to(device)
    start_time = time.time()

    print('BATCHSIZE:', args.batchsize)
    train_loader, vali_loader, test_loader = create_datloaders(batchsize=args.batchsize, dataset=vqvae_config["dataset"], base_path=args.base_path)

    for epoch in range(int((vqvae_config['num_training_updates']/len(train_loader)) + 0.5)):
        model.train()
        for i, (batch_x) in enumerate(train_loader):
            tensor_all_data_in_batch = torch.tensor(batch_x, dtype=torch.float, device=device)

            if logger is not None:
                loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, encoding_indices, encodings = \
                    model.shared_eval(tensor_all_data_in_batch, optimizer, 'train', comet_logger=logger)
            else:
                loss, vq_loss, recon_error, x_recon, perplexity, embedding_weight, encoding_indices, encodings = \
                    model.shared_eval(tensor_all_data_in_batch, optimizer, 'train')

        if epoch % 1000 == 0:
            torch.save(model, os.path.join(save_dir, f'checkpoints/model_epoch_{epoch}.pth'))
            print('Saved model from epoch ', epoch)

    print('total time: ', round(time.time() - start_time, 3))
    return model


def create_datloaders(batchsize=100, dataset="dummy", base_path='dummy'):
    supported_datasets = ['msl', 'psm', 'smap', 'smd', 'swat', 'all', 'odometry']
    if dataset not in supported_datasets:
        raise ValueError(f"Unknown dataset '{dataset}' - update create_datloaders to support it.")

    full_path = base_path

    if dataset in ['msl', 'psm', 'smap', 'smd', 'swat']:
        train_data = np.load(os.path.join(full_path, "train.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test.npy"), allow_pickle=True)
        val_data = None

        train_data = np.swapaxes(train_data, 1, 2).reshape(-1, train_data.shape[-1])
        test_data = np.swapaxes(test_data, 1, 2).reshape(-1, test_data.shape[-1])

    else:
        train_data = np.load(os.path.join(full_path, "train_data_x.npy"), allow_pickle=True)
        val_data = np.load(os.path.join(full_path, "val_data_x.npy"), allow_pickle=True)
        test_data = np.load(os.path.join(full_path, "test_data_x.npy"), allow_pickle=True)

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=batchsize,
                                                   shuffle=True,
                                                   num_workers=1,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=1,
                                                drop_last=False) if val_data is not None else None

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batchsize,
                                                shuffle=False,
                                                num_workers=1,
                                                drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='', help='path to config file')
    parser.add_argument('--model_init_num_gpus', type=int, default=0, help='GPU id')
    parser.add_argument('--data_init_cpu_or_gpu', type=str, help='Data location')
    parser.add_argument('--comet_log', action='store_true', help='Log to Comet')
    parser.add_argument('--comet_tag', type=str, help='Comet tag')
    parser.add_argument('--comet_name', type=str, help='Comet name')
    parser.add_argument('--save_path', type=str, help='Checkpoint save path')
    parser.add_argument('--base_path', type=str, default=False, help='Base data path')
    parser.add_argument('--batchsize', type=int, required=True, help='Batch size')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')

    args = parser.parse_args()

    config_file = args.config_path
    print('Config folder:\t {}'.format(config_file))

    with open(config_file, 'r') as f:
        config = json.load(f)
    print(' Running Config:', config_file)

    save_folder_name = ('CD' + str(config['vqvae_config']['embedding_dim']) +
                        '_CW' + str(config['vqvae_config']['num_embeddings']) +
                        '_CF' + str(config['vqvae_config']['compression_factor']) +
                        '_BS' + str(args.batchsize) +
                        '_ITR' + str(config['vqvae_config']['num_training_updates']) +
                        '_seed' + str(args.seed))

    save_dir = args.save_path + save_folder_name

    master = {
        'start_time': strftime("%Y-%m-%dT%H-%M-%S", gmtime()),
        'config file': config_file,
        'save directory': save_dir,
        'gpus': args.model_init_num_gpus,
    }

    if args.comet_log:
        try:
            comet_logger = comet_ml.Experiment(
                api_key=config['comet_config']['api_key'],
                project_name=config['comet_config']['project_name'],
                workspace=config['comet_config']['workspace'],
            )
            comet_logger.add_tag(args.comet_tag)
            comet_logger.set_name(args.comet_name)
        except Exception as e:
            print(f"❌ Failed to initialize comet: {e}. Continuing without comet logging.")
            comet_logger = None
    else:
        comet_logger = None

    if torch.cuda.is_available() and args.model_init_num_gpus >= 0:
        assert args.model_init_num_gpus < torch.cuda.device_count()
        device = 'cuda:{:d}'.format(args.model_init_num_gpus)
    else:
        device = 'cpu'

    data_init_loc = device if args.data_init_cpu_or_gpu == 'gpu' else 'cpu'

    main(device, config, save_dir, comet_logger, data_init_loc, args)
