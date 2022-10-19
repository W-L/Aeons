from time import sleep
import aeons.aeons_core

parser = aeons.aeons_core.setup_parser()
args = parser.parse_args()


#%%
# DEBUG MODE
# import os
# wd = "/home/lukas/Desktop/Aeons/18_live"
# param_file = "../params/live/zymolog_live.params"
# os.chdir(wd)
# from aeons.aeons_utils import read_args_fromfile
# args = read_args_fromfile(parser=parser, file=param_file)
#%%

args.live = 1

ar = aeons.aeons_core.AeonsRun(args=args)

# initialise main loop - periodically check for new data
try:
    while True:
        next_update = ar.process_batch_live()
        # if processing was faster than the waiting time, sleep the rest
        if next_update > 0:
            sleep(next_update)


except KeyboardInterrupt:
    print("exiting after keyboard interrupt.. ")
