import sys
sys.path.insert(0, "/home/lukas/software/longreadsim")
sys.path.insert(0, "/nfs/research1/goldman/lukasw/Aeons/lrs")
sys.path.insert(0, "/nfs/research1/goldman/lukasw/Aeons/code")

import longreadsim
import os
import Aeons
# import numpy as np
# from importlib import reload
# reload(Aeons)
# reload(longreadsim)
os.chdir("/home/lukas/Desktop/Aeons/28_profile2")
# os.chdir("/nfs/research1/goldman/lukasw/Aeons/01_assembly")
cwd = os.getcwd()

#%%
# vars for simulated seqs
N = int(1e4)
mu = 200
lam = 4000
sd = 1000
k = 23
err = 0.01
maxbatch = 20
batchsize = 10
size = int(1e7)

const = Aeons.Constants(mu=mu, lam=lam, sd=sd, N=N, k=k, err=err, maxbatch=maxbatch, batchsize=batchsize, size=size)
#%%
# simulate a genome and a single fastq that can be used to mmap
genome = longreadsim.SimGenome([N], perc=const.perc)
read_dist = longreadsim.ReadDist(N=N, lam=lam, sd=sd, mu=mu)
fastq = longreadsim.SimFastq(genome=genome, read_dist=read_dist, batch_size=2000, err=const.err, errD=const.errD, write=True)
reads = fastq.simulate_batch()
#%%
# set up AeonsRun
ar = AeonsRun(const=const, fq_source=f'{cwd}/reads_0.fastq')
# ar = AeonsRun(const=const, fq_source=f'{cwd}/reads_0.fastq')
self = ar

for i in range(maxbatch):
    ar.process_batch()






