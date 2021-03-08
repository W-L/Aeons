import longreadsim
import numpy as np
import Aeons
from Aeons import plot_gt
from importlib import reload
reload(Aeons)
reload(longreadsim)

"""
this is aeons with simulated reads - just construction and population
"""

#%%
# vars for simulated seqs
N = 2_00
mu = 2
lam = 30
sd = 1
batch_size = 10
err = 0.01
errD = 0.0

# vars for aeons
k = 13
prior = 0.01

rho = 10
alpha = 10

# perc=0.0
#%%
# simulate a genome and some reads
genome = longreadsim.SimGenome([N])
read_dist = longreadsim.ReadDist(N=N, lam=lam, sd=sd, mu=mu)
fastq = longreadsim.SimFastq(genome=genome, read_dist=read_dist, batch_size=batch_size, err=err, errD=errD)
reads = fastq.simulate_batch()
#%%
# initialise bloom filter, red length dist and assembly
bloom = Bloom(k=k, genome_estimate=N)
rld = LengthDist(lam=lam, sd=sd, mu=mu)
dbg = SparseGraph(size=int(1e7), bloom=bloom, rld=rld)

#%%

bloom.fill(reads=reads)
rld.record(reads=reads)

dbg.update_graph(updated_kmers=dbg.bloom.updated_kmers)

self=dbg
dbg.reduce_matrix()
dbg.add_absorbers()
dbg.gt_format()
#%%
plot_s(dbg.gtg)

graph_draw(dbg.gtg, vertex_fill_color=dbg.gtg.vp.scores, vcmap=cm.coolwarm, vertex_text=dbg.gtg.vp.ind)

#%%
from time import sleep
for i in range(5):
    reads = fastq.simulate_batch()
    dbg.bloom.fill(reads=reads)
    dbg.update_graph(updated_kmers=dbg.bloom.updated_kmers)
    dbg.gt_format()
# sleep(1)
# print("--")
    # print("--")

self=dbg
graph_draw(dbg.gtg, vertex_fill_color=dbg.gtg.vp.scores)
plot_s(dbg.gtg)
#%%

ass = Assembly(bloom=bloom)
ass.reconstruct()

# ass.bloom.legacy_fill(reads=reads)
# ass.reconstruct()


graph_draw(ass.dbg)

