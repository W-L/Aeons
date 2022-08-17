import aeons.aeons_core

parser = aeons.aeons_core.setup_parser()
args = parser.parse_args()

#%%
ar = aeons.aeons_core.AeonsRun(args=args)

while ar.batch <= args.maxb:
    ar.process_batch()
