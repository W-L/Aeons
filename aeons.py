import aeons.aeons_core
from aeons.aeons_config import load_config
from time import sleep

# this is for debugging - not using CL parser
# from argparse import Namespace
# import os
# print("\nDEBUG ACTIVE\n")
# os.chdir("/home/lukas/Desktop/Aeons/44_toml")
# toml_paths = Namespace(toml="/home/lukas/Desktop/Aeons/params/local/test_aeons_sim.toml",
#                        toml_readfish="/home/lukas/Desktop/Aeons/params/local/test_readfish.toml")
# args = load_config(toml_paths=toml_paths)

# non-debug
args = load_config()

#%%

ar = aeons.aeons_core.AeonsRun(args=args)

if args.sim_run:
    while ar.batch <= args.maxb:
        ar.process_batch_sim()

    ar.cleanup()

    with open("aeons.done", 'w') as adone:
        adone.write('')

elif args.live_run:
    try:
        while True:
            next_update = ar.process_batch_live()
            # if processing was faster than the waiting time, sleep the rest
            if next_update > 0:
                sleep(next_update)


    except KeyboardInterrupt:
        print("exiting after keyboard interrupt.. ")


