def main_worker(args):
    
    args.class_list = distribute_class(args.nclass,args.debug)

    plotter = get_plotter(args)

    loader_real,_ = get_loader(args)


    aug, _ = diffaug(args)
    
    # Condenser 使用 NCFM 当前分布式设备；CPU/Gloo smoke 不能硬编码 CUDA。
    condenser = Condenser(
        args,
        nclass_list=args.class_list,
        nchannel=args.nch,
        hs=args.size,
        ws=args.size,
        device=args.device,
    )
    for local_rank in range(args.local_world_size):
        if  args.local_rank == local_rank:
            condenser.load_condensed_data(loader_real, init_type=args.init,load_path=args.load_path)
            print(f"============RANK:{dist.get_rank()}====LOCAL_RANK {local_rank} Loaded Condensed Data==========================")
        dist.barrier()

    optim_img = get_optimizer(optimizer=args.optimizer, parameters=condenser.parameters(),lr=args.lr_img, mom_img=args.mom_img,weight_decay=args.weight_decay,logger=args.logger)
    model_init,model_interval,model_final = get_feature_extractor(args)
    if args.sampling_net:
        # The flattened ConvNet feature dimension depends on input size and
        # depth.  The old hard-coded 2048 only worked for some 32/128px
        # configurations and failed for COVID (112px).  Infer it from the
        # actual model used by the condenser and generate exactly num_freqs
        # frequencies for a fair learned-frequency comparison.
        model_interval.eval()
        with torch.no_grad():
            probe = torch.zeros(
                2, args.nch, args.size, args.size, device=args.device
            )
            _, probe_features = model_interval(probe, return_features=True)
        feature_dim = int(probe_features.shape[1])
        sampling_net = SampleNet(
            feature_dim=feature_dim,
            t_batchsize=int(args.num_freqs),
        ).to(args.device)
        # The synthetic images are partitioned by class across ranks, but the
        # learned frequency proposal is one global model. Without DDP, each
        # rank would optimize an independent proposal and rank 0's artifact
        # would silently mix unsynchronized frequency networks.
        sampling_net = DDP(
            sampling_net,
            device_ids=[args.local_rank] if args.device.type == "cuda" else None,
            broadcast_buffers=True,
        )
        if args.rank == 0:
            args.logger(
                f"SamplingNet feature_dim={feature_dim}, num_freqs={args.num_freqs}"
            )
        optim_sampling_net = get_optimizer(
            optimizer="sgd",
            parameters=sampling_net.parameters(),
            lr=args.lr_sampling_net,
            mom_img=args.mom_img,
            weight_decay=args.weight_decay,
            logger=args.logger,
        )
    else:
        sampling_net = None
        optim_sampling_net = None
    condenser.condense(args,plotter,loader_real,aug,optim_img,model_init,model_interval,model_final,sampling_net,optim_sampling_net)

    dist.destroy_process_group()



if __name__ == '__main__':
    import sys
    import os
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.diffaug import diffaug
    import torch.distributed as dist
    from  utils.ddp import distribute_class
    from  utils.utils import get_plotter,get_optimizer,get_loader,get_feature_extractor
    from utils.init_script import init_script
    import argparse
    from argsprocessor.args import ArgsProcessor
    from condenser.Condenser import Condenser
    from NCFM.SampleNet import SampleNet

    parser = argparse.ArgumentParser(description='Configuration parser')
    parser.add_argument('--debug',dest='debug',action='store_true',help='When dataset is very large , you should get it')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--run_mode',type=str,choices=['Condense', 'Evaluation',"Pretrain"],default='Condense',help='Condense or Evaluation')
    parser.add_argument('-a','--aug_type',type=str,default='color_crop_cutout',help='augmentation strategy for condensation matching objective')
    parser.add_argument('--init',type=str,default='mix',choices=['random', 'noise', 'mix', 'load'],help='condensed data initialization type')
    parser.add_argument('--load_path',type=str,default=None,help="Path to load the synset")
    parser.add_argument('--gpu', type=str, default = "0",required=True, help='GPUs to use, e.g., "0,1,2,3"') 
    parser.add_argument('-i', '--ipc', type=int, default=1,required=True, help='number of condensed data per class')
    parser.add_argument('--tf32', action='store_true',default=True,help='Enable TF32')
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.config_path)

    args = args_processor.add_args_from_yaml(args)

    # Variant jobs use explicit environment overrides while the released
    # baseline YAML remains unchanged and reproducible.
    if "NCFM_SAMPLING_NET" in os.environ:
        args.sampling_net = os.environ["NCFM_SAMPLING_NET"].strip().lower() in {
            "1", "true", "yes"
        }
    if "NCFM_FREQUENCY_SAMPLER" in os.environ:
        args.frequency_sampler = os.environ["NCFM_FREQUENCY_SAMPLER"].strip().lower()
    if "NCFM_IMPORTANCE_MEAN_SHIFT" in os.environ:
        args.importance_mean_shift = float(os.environ["NCFM_IMPORTANCE_MEAN_SHIFT"])
    if "NCFM_FREQUENCY_SEED" in os.environ:
        args.frequency_seed = int(os.environ["NCFM_FREQUENCY_SEED"])
    if "NCFM_CONDENSE_SEED" in os.environ:
        args.seed = int(os.environ["NCFM_CONDENSE_SEED"])
    if "NCFM_OBJECTIVE" in os.environ:
        args.objective = os.environ["NCFM_OBJECTIVE"].strip().lower()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    init_script(args)

    main_worker(args)
