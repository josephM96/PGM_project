import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=1
    )

    # build model architecture
    model_background = config.init_obj('arch_background', module_arch)
    model_semantic = config.init_obj('arch_semantic', module_arch)

    logger.info(model_background)
    logger.info(model_semantic)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model_background)
    model_background.load_state_dict(state_dict)
    model_semantic.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_b = model_background.to(device)
    model_s = model_semantic.to(device)
    model_b.eval()
    model_s.eval()

    total_loss = 0.0

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output_b, output_s = model_b(data), model_s(data)

            # computing loss, metrics on test set
            loss_b = loss_fn(output_b, target)
            loss_s = loss_fn(output_s, target)

            batch_size = data.shape[0]
            total_loss += (loss_s.item() - loss_b.item()) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
