import sys
sys.path.insert(0, "/home/lukas/software/longreadsim")
import longreadsim
import os
import numpy as np
# import Aeons
# from Aeons import plot_gt
# from importlib import reload
# reload(Aeons)
# reload(longreadsim)
os.chdir("/home/lukas/Desktop/Aeons/26_structures")
cwd = os.getcwd()

#%%
# construct a perfectly linear graph first
N = 2000
perc = 0.0
lam = 2000
sd = 1
mu = 1
bs = 20
err = 0.0
errD = 0.0
genome = longreadsim.SimGenome([N], perc=perc)
read_dist = longreadsim.ReadDist(N=N, lam=lam, sd=sd, mu=mu)
fastq = longreadsim.SimFastq(genome=genome, read_dist=read_dist, batch_size=bs, err=err, errD=errD, write=True)
reads = fastq.simulate_batch()


#%%
N = 3000
mu = 3
lam = 35
# lam = 50
sd = 5
# sd = 8

k = 23
err = 0.00

maxbatch = 10
batchsize = 1

size = int(1e7)
const = Constants(mu=mu, lam=lam, sd=sd, N=N, k=k, err=err, maxbatch=maxbatch, batchsize=batchsize, size=size)

#%%
# FIRST A LOOSE END CAUSE BY SNP AT END
ar = AeonsRun(const=const, fq_source=f'{cwd}/reads_snp.fastq')
self = ar

ar.process_batch2()



read_sequences = ar.fq.get_batch()

# introduce deliberate error at end of read
# get id
rid = list(read_sequences.keys())[0]
seq = read_sequences[rid][0:100]
len(seq)
seq = seq[: -11] + "A" + seq[-10: ]
# trim the read so that it just about adds the error kmers
# to the large component but no additional kmers
seq = seq[-(int(k*1.5)):]
len(seq)

# add to graph
reads = {rid: seq}
# ar.process_batch(reads=reads)

ar.process_batch2(reads=reads)

#%%
self.dbg.gt_format()
plot_index(graph=self.dbg.gtg, name="indices")
#
plot_scores(graph=self.dbg.gtg, name="scores")

self.dbg.gt_format(mat=self.dbg.benefit)
plot_benefit(graph=self.dbg.gtg, name="benefit")

# self.dbg.gt_format(mat=self.dbg.benefit_raw)
# plot_benefit(graph=self.dbg.gtg, name="benefit_raw")


plot_complex(dbg=ar.dbg, totalsteps=120, junctiondist=50, name="loose", start=None, layout=None)


#%%
# NEXT CREATE AN EXAMPLE OF A SIMPLE BUBBLE
mu = 2
lam = 50
sd = 5

k = 23
err = 0.00

maxbatch = 10
batchsize = 1

size = int(1e7)
const = Constants(mu=mu, lam=lam, sd=sd, N=N, k=k, err=err, maxbatch=maxbatch, batchsize=batchsize, size=size)

ar = AeonsRun(const=const, fq_source=f'{cwd}/reads_snp.fastq')
# ar = AeonsRun(const=const, fq_source=f'{cwd}/reads_mini.fastq')
self = ar
ar.process_batch2()
#%%

read_sequences = ar.fq.get_batch()

# introduce error inside of read
rid = list(read_sequences.keys())[0]
seq = read_sequences[rid][0:100]
len(seq)
seq = seq[: 50] + "A" + seq[51: ]
# trim the read so that it just about adds the error kmers
# to the large component but no additional kmers
# seq = seq[-(int(k*1.5)):]
len(seq)

# add to graph
reads = {rid: seq}


ar.process_batch2(reads=reads)





#%%
self.dbg.gt_format()
plot_index(graph=self.dbg.gtg, name="indices", vsize=30)
#
plot_scores(graph=self.dbg.gtg, name="scores")
#
self.dbg.gt_format(mat=self.dbg.benefit)
plot_benefit(graph=self.dbg.gtg, name="benefit")


plot_complex(dbg=ar.dbg, totalsteps=150, junctiondist=65, name="bubble", start=4202515, layout=None)

#%%

# EXAMPLE OF A more complex bubble
mu = 2
lam = 65
sd = 20

k = 23
err = 0.00

maxbatch = 10
batchsize = 1

size = int(1e7)
const = Constants(mu=mu, lam=lam, sd=sd, N=N, k=k, err=err, maxbatch=maxbatch, batchsize=batchsize, size=size)

ar = AeonsRun(const=const, fq_source=f'{cwd}/reads_snp.fastq')
self = ar
ar.process_batch2()


read_sequences = ar.fq.get_batch()

rid = list(read_sequences.keys())[0]
oseq = read_sequences[rid][0:100]
len(oseq)

seqlist = list(oseq)
mutclust = 65
seqlist[mutclust] = 'C'
seq = ''.join(seqlist)
len(seq)
# add to graph
reads = {rid: seq}
ar.process_batch2(reads=reads, no_incr=True)

seqlist = list(oseq)
seqlist[mutclust + 1] = 'T'
seq = ''.join(seqlist)
len(seq)
# add to graph
reads = {rid: seq}
ar.process_batch2(reads=reads, no_incr=True)

seqlist = list(oseq)
seqlist[mutclust + 3] = 'C'
seq = ''.join(seqlist)
len(seq)
# add to graph
reads = {rid: seq}
ar.process_batch2(reads=reads, no_incr=True)

seqlist = list(oseq)
seqlist.pop(mutclust + 5)
seq = ''.join(seqlist)
len(seq)
# add to graph
reads = {rid: seq}
ar.process_batch2(reads=reads, no_incr=True)

seqlist = list(oseq)
seqlist[mutclust + 15] = 'G'
seq = ''.join(seqlist)
len(seq)
# add to graph
reads = {rid: seq}
ar.process_batch2(reads=reads, no_incr=True)




#%%
self.dbg.gt_format()
# plot_index(graph=self.dbg.gtg, name="indices.pdf")

plot_scores(graph=self.dbg.gtg, name="scores")
#
self.dbg.gt_format(mat=self.dbg.benefit)
plot_benefit(graph=self.dbg.gtg, name="benefit")


plot_complex(dbg=ar.dbg, totalsteps=270, junctiondist=74, name="tangle", start=1200753, layout=None)




