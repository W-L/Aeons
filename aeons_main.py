import aeons.aeons_core
from importlib import reload
reload(aeons.aeons_core)
from aeons.aeons_utils import read_args_fromfile

#%%
# DEBUG MODE
import os
wd = "/home/lukas/Desktop/Aeons/15_zymolog"
param_file = "../params/local/zymolog_testing.params"
os.chdir(wd)
parser = aeons.aeons_core.setup_parser()
args = read_args_fromfile(parser=parser, file=param_file)


#%%

ar = aeons.aeons_core.AeonsRun(args=args)
self = ar

#%%

while ar.batch <= args.maxb:
    ar.process_batch()

ar.cleanup()

#%% auto snakemake
# if args.snake:
#     from aeons.aeons_utils import launch_post_snake
#     launch_post_snake(run_name=args.name, configfile=args.snake)
