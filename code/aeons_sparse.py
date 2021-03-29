import sys
sys.path.insert(0, "/home/lukas/software/longreadsim")

import longreadsim
import os
import numpy as np
import Aeons
# from Aeons import plot_gt
# from importlib import reload
# reload(Aeons)
# reload(longreadsim)
os.chdir("/home/lukas/Desktop/Aeons/24_gfa")
cwd = os.getcwd()

#%%
# vars for simulated seqs
N = 30000
mu = 30
lam = 500
sd = 10

k = 23
err = 0.02

maxbatch = 10
batchsize = 10

size = int(1e7)

const = Aeons.Constants(mu=mu, lam=lam, sd=sd, N=N, k=k, err=err, maxbatch=maxbatch, batchsize=batchsize, size=size)
#%%
# simulate a genome and a single fastq that can be used to mmap
genome = longreadsim.SimGenome([N], perc=const.perc)
read_dist = longreadsim.ReadDist(N=N, lam=lam, sd=sd, mu=mu)
fastq = longreadsim.SimFastq(genome=genome, read_dist=read_dist, batch_size=1000, err=const.err, errD=const.errD, write=True)
reads = fastq.simulate_batch()
#%%
# set up AeonsRun
ar = Aeons.AeonsRun(const=const, fq_source=f'{cwd}/reads_0.fastq')
self = ar

ar.process_batch()


#%%
# ar.dbg.gt_format(mat=ar.dbg.benefit)
# plot_w(ar.dbg.gtg)
#%%
# TODO fix (maybe create gtg again within)
# plot_complex(dbg=self.dbg, steps=100)

#%%
from time import sleep
# for i in range(5):
#     reads = fastq.simulate_batch()
#     dbg.process_batch(reads)
    # dbg.gt_format()

# self=dbg

