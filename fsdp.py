import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from yacs.config import CfgNode as CN

from lib.HRNet import get_FOF
from lib.dataset import TMPP
from lib.utils import toDevice

def setup(rank, world_size):
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(rank)
    if world_size==1: return
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12375'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup(world_size):
    if world_size==1: return
    dist.destroy_process_group()

def train(args, model, rank, epoch, DL, optimizer, scaler, sampler=None):
    model.train()
    if sampler:
        sampler.set_epoch(epoch)
    r0_loss = None

    for it,data in enumerate(DL):
        data = toDevice(data, rank)
        if args.amp:
            with torch.autocast("cuda"):
                loss = F.mse_loss(data["fof"], model(data["img"]))
                lbt = {"mse":loss.item()}
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = F.mse_loss(data["fof"], model(data["img"]))
            lbt = {"mse":loss.item()}
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # gather loss
        if args.lbt and rank==0:
            if r0_loss is None:
                r0_loss = lbt
            else:
                for k,v in lbt.items():
                    r0_loss[k] += v
            if (it+1)%args.fre==0:
                info = "["+time.strftime("%m.%d-%H:%M:%S", time.localtime())+"]\t"
                info += "Epoch %03d, iteration %08d, " % (epoch+1, it+1)
                for k,v in r0_loss.items():
                    info += "\t%s:%.6f" % (k,v/args.fre)
                print(info)
                r0_loss = None
       
def main(rank, world_size, args):
    setup(rank, world_size)

    # dataset
    dataset = TMPP()
    sampler = None if world_size==1 else DistributedSampler(dataset)
    DL = torch.utils.data.DataLoader(
        dataset, batch_size=args.bs//world_size, sampler=sampler,
        shuffle=True if world_size==1 else None,
        num_workers=2, pin_memory=True, persistent_workers=True,
        worker_init_fn=dataset.open_db)
    if rank == 0:
        print("Dataloader size:", len(DL))
    args.fre = len(DL)//args.fre_num

    # model & others
    start_epoch = 0
    model = get_FOF().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()

    # load
    os.makedirs("ckpt/%s"%args.name, exist_ok=True)
    ckpt_list = sorted(os.listdir(os.path.join("ckpt", args.name)))
    if len(ckpt_list) > 0:
        ckpt_path = os.path.join("ckpt", args.name, ckpt_list[-1])
        if rank == 0:
            print('Resuming from ', ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        start_epoch = state_dict["epoch"]
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scaler.load_state_dict(state_dict["scaler"])
        del state_dict

    if world_size != 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model)

    for epoch in range(start_epoch,1000,1):
        # train one epoch
        train(args, model, rank, epoch, DL, optimizer, scaler, sampler=sampler)
        
        # save model
        dist.barrier()
        if rank == 0:
            torch.save({
                "epoch" : epoch+1,
                "model" : model.state_dict() if world_size==1 else model.module.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scaler" : scaler.state_dict()
            },"ckpt/%s/%03d.pth"%(args.name,epoch+1))
    
    cleanup(world_size)

if __name__ == '__main__':
    #-------------cfg here-------------
    # name amp fre
    args = CN()
    args.name = "tmp"
    args.amp = True
    args.fre_num = 4
    args.device = "0,1"
    args.lbt = True
    args.bs = 32
    #----------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    WORLD_SIZE = torch.cuda.device_count()
    if WORLD_SIZE==1:
        main(0,1,args)
    else:
        mp.spawn(main,
            args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,
            join=True)
