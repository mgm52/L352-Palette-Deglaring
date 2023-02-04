import argparse
import glob
import os
import sys
import traceback
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
import wandb

def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
        
def main(opt):
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)

def main_with_wandb():
    with wandb.init() as run:
        wcfg = wandb.config

        opt = Praser.parse(args)
        wcfg = wandb.config.as_dict()

        opt['datasets']['train']['dataloader']['args']['batch_size'] = wcfg['batch_size']
        opt['model']['which_model']['args']['optimizers'][0]['lr'] = wcfg['start_lr']
        opt['model']['which_model']['args']['optimizers'][0]['weight_decay'] = wcfg['weight_decay']
        for u_p in ['inner_channel', 'channel_mults', 'attn_res', 'num_head_channels', 'res_blocks', 'dropout', 'groupnorm']:
            opt['model']['which_networks'][0]['args']['unet'][u_p] = wcfg['unet_' + u_p]
        opt['model']['which_networks'][0]['args']['beta_schedule']['train']['schedule'] = wcfg['beta_schedule']

        opt['train']['val_epoch'] = 5
        opt['train']['n_epoch'] = 5
        opt['train']['save_checkpoint_epoch'] = 999 # save ssd space

        opt['model']['which_model']['args']['quiet'] = True

        try:
            main(opt)
        except Exception as e:
            raise Exception("Error in main_with_wandb: {}, stack trace: {}".format(e, traceback.format_exc()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('-rr', '--runresume', type=str, default=None)
    parser.add_argument('-c', '--config', type=str, default='config/glareremoval.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-sid', '--sweep_id', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ws', '--wandb_sweep', action='store_true')
    parser.add_argument('-wr', '--wandb_run', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-bs', '--batchsize_mult', type=str, default=None)
    parser.add_argument('-lr', '--learningrate_mult', type=str, default=None)
    parser.add_argument('-sub', '--forcesub', action='store_true')

    args = parser.parse_args()

    #args.wandb_run = True

    if args.wandb_run:
        if args.runresume:
            wandb.init(project='PaletteDeglare-LongRuns', id=args.runresume, resume="must")
            print(f"Successfully resumed run {args.runresume}")
        else:
            wandb.init(project='PaletteDeglare-LongRuns')
        wcfg = wandb.config
        #wcfg.update(args)
        opt = Praser.parse(args)
        if args.batchsize_mult is not None:
            opt['datasets']['train']['dataloader']['args']['batch_size'] = int(opt['datasets']['train']['dataloader']['args']['batch_size'] * float(args.batchsize_mult))
            print(f"Setting batchsize to {opt['datasets']['train']['dataloader']['args']['batch_size']}")
        if args.learningrate_mult is not None:
            opt['model']['which_model']['args']['optimizers'][0]['lr'] *= float(args.learningrate_mult)
            print(f"Setting learningrate to {opt['model']['which_model']['args']['optimizers'][0]['lr']}")
        if args.forcesub:
            opt['datasets']['train']['which_dataset']['args']['gt_is_flare_diff'] = True
            print(f"Setting gt_is_flare_diff to True")
        if args.resume:
            if not "/" in args.resume:
                # assume args.resume is in format train_glareremoval_230124_001715
                args.resume = f"experiments/{args.resume}/checkpoint"
                # Find the highest-numbered checkpoint
                checkpoints = glob.glob(f"{args.resume}/*.state")
                if len(checkpoints) > 0:
                    checkpoints = [int(os.path.splitext(os.path.basename(checkpoint))[0]) for checkpoint in checkpoints]
                    args.resume = f"{args.resume}/{max(checkpoints)}"
                    print("Found checkpoint: {}".format(args.resume))
            opt['path']['resume_state'] = args.resume
            print(f"Going to resume from checkpoint {args.resume}")

        wcfg.update(opt, allow_val_change=True)
        try:
            main(opt)
        except Exception as e:
            raise Exception("Error in main_with_wandb: {}, stack trace: {}".format(e, traceback.format_exc()))
        
    elif args.wandb_sweep:
        sweep_configuration = {
            'method': 'bayes',
            'name': 'sweep',
            'metric': {
                'goal': 'minimize', 
                'name': 'val/mae'
                },
            'parameters': {
                'batch_size': {'values': [64]},
                'start_lr': {'max': 5e-3, 'min': 5e-6, 'distribution': 'log_uniform_values'},
                'weight_decay': {'values': [1e-5, 1e-6, 1e-7, 0.0]},
                'unet_inner_channel': {'values': [8, 16, 32, 64]},
                'unet_channel_mults': {'values': [[1, 2], [1, 2, 4], [1, 2, 4, 8]]},
                'unet_attn_res': {'values': [[8], [16]]},
                'unet_num_head_channels': {'values': [8, 16, 32, 64]},
                'unet_res_blocks': {'values': [1, 2, 3]},
                'unet_dropout': {'values': [0.1, 0.2]},
                'unet_groupnorm': {'values': [False]},          # <- Keeping groupnorm off due to consistently bad results
                'beta_schedule': {'values': ["linear", "quad"]}
            }
        }

        if args.sweep_id is not None:
            sweep_id = args.sweep_id
        else:
            sweep_id = wandb.sweep(sweep=sweep_configuration, project="PaletteDeglare-Sweeps")
        wandb.agent(sweep_id=sweep_id, function=main_with_wandb, project="PaletteDeglare-Sweeps")
    else:
        ''' parser configs '''
        # args.debug = True
        opt = Praser.parse(args)
        main(opt)