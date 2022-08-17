import Aeons
from time import sleep
import logging
#%%
# set up arguments
parser = Aeons.setup_parser()
args = parser.parse_args()
#%%
const = Aeons.Constants()
ar = Aeons.AeonsRun(args=args, const=const)
logging.info("starting initialisation")


#%%
# initialise main loop - periodically check for new data
logging.info('Initialisation completed.. waiting for sequencing data\n')
try:
    while True:
        next_update = ar.process_batch_live()
        # if processing was faster than the waiting time, sleep the rest
        if next_update > 0:
            sleep(next_update)


except KeyboardInterrupt:
    logging.info("exiting after keyboard interrupt.. ")



