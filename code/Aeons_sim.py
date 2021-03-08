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
N = 1_00
mu = 20
lam = 40
sd = 10
batch_size = 200
err = 0.00
errD = 0.0

# vars for aeons
k = 13
prior = 0.01

rho = 10
alpha = 10

# perc=0.0
#%%
# simulate a genome and some reads
# genome = longreadsim.SimGenome([N])
# read_dist = longreadsim.ReadDist(N=N, lam=lam, sd=sd, mu=mu)
# fastq = longreadsim.SimFastq(genome=genome, read_dist=read_dist, batch_size=batch_size, err=err, errD=errD)
# reads = fastq.simulate_batch()
#%%
# initialise bloom filter and load it with the first batch
# bloom = Aeons.Bloom(k=k, genome_estimate=N)

# init assembly
# dbg = Aeons.Assembly(bloom=bloom)
# dbg.initial_construction(reads=reads)
# self = dbg

# init properties
# dbg.init_prior(prior=prior)
# dbg.init_scores()
# dbg.init_utility()
# dbg.init_smu()

# TODO fix
# dbg.init_timeCost(lam=lam, mu=mu, rho=rho, alpha=alpha)



#%%
# update the assembly with new reads
# from time import sleep
# for i in range(30):
#     new_reads = fastq.simulate_batch()
#     dbg.update(new_reads=new_reads)
#     sleep(1)
#%%


# plot_gt(dbg.dbg, comp=True)


#%%

# update the scores
# dbg.update_scores()  # uses the bloom p1

# scores = Aeons.updateS(graph=dbg, bloom_paths=bloom_reads_plusone, t_solid=t_solid)

# update U - improvised length dist
# ldist = Aeons.trunc_normal(mu=mu, sd=sd, lam=lam)
# ccl = Aeons.comp_cml_dist(ldist)

utility = Aeons.updateU(graph=dbg, ccl=ccl, mu=mu, bloom_paths=bloom_reads_plusone)

# strat = find_strat_aeons(graph=dbg)


#%%
# dbg = gt.topology.extract_largest_component(dbg)
# Aeons.plot_gt(dbg, comp=True)
Aeons.plot_gt(dbg, vcolor=dbg.vp.scores, ecolor=dbg.ep.edge_weights)
Aeons.plot_benefit(dbg, 1)  # TODO check why the benefit is distributed weirdly


#%%
# create a bigger assembly and only viz a small part
#
# viz_range = list(range(0,400))
# viz = dbg.new_vertex_property("bool")
# viz.a[viz_range] = 1
#
# dbg_viz = gt.GraphView(dbg, vfilt=viz)
#






