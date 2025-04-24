import os
import subprocess
from collections import namedtuple

fake_idrs_torch = namedtuple('fake_idrs_torch', ['size', 'rank', 'local_rank'])

def init_distributed_mode():
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(gpu)
        os.environ['WORLD_SIZE'] = str(world_size)

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr

        return fake_idrs_torch(size=world_size, rank=rank, local_rank=gpu)

    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        return fake_idrs_torch(size=world_size, rank=rank, local_rank=gpu)

    else:
        print('Not using distributed mode')
        return None
