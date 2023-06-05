import aeons.aeons_core

parser = aeons.aeons_core.setup_parser()
args = parser.parse_args()

#%%
ar = aeons.aeons_core.AeonsRun(args=args)

while ar.batch <= args.maxb:
    ar.process_batch()

ar.cleanup()

#%%
with open("aeons.done", 'w') as adone:
    adone.write('')


#%% auto snakemake
# if args.snake:
#     from aeons.aeons_utils import launch_post_snake
#     launch_post_snake(run_name=args.name, configfile=args.snake)

