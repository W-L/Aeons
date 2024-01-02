import argparse
from argparse import Namespace
from pathlib import Path
import logging
import sys

import rtoml

from aeons_utils import execute, find_exe, init_logger

"""
Configuration of Aeons and Readfish:

1. load a template toml that contains defaults
2. parse a toml given on command line
3. overwrite defaults with CL toml
4. parse a toml for readfish separately
5. exchange args between the two
"""


class Config:
    def __init__(self):
        self.template_toml = Path(sys.argv[0]).parent / "aeons_template.toml"
        if not self.template_toml.is_file():
            raise FileNotFoundError("Template TOML with configuration defaults not found. Exiting")


    def load_defaults(self):
        self.args = rtoml.load(self.template_toml)


    @staticmethod
    def get_toml_paths():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--toml',
            type=str,
            default=None,
            required=True,
            help='TOML configuration file')
        parser.add_argument(
            '--toml_readfish',
            type=str,
            default=None,
            help='TOML configuration file for readfish')
        toml_paths = parser.parse_args()
        return toml_paths


    @staticmethod
    def load_toml_args(toml_paths):
        args_aeons_file = rtoml.loads(Path(toml_paths.toml).read_text(encoding="utf-8"))
        if toml_paths.toml_readfish:
            args_readfish_file = rtoml.loads(Path(toml_paths.toml_readfish).read_text(encoding="utf-8"))
        else:
            args_readfish_file = dict()
        return args_aeons_file, args_readfish_file


    def overwrite_defaults(self, args_from_file):
        # overwrite defaults
        for category in args_from_file:
            for k, v in args_from_file[category].items():
                self.args[category][k] = v


    def convert_to_namespace(self):
        args = Namespace()
        for category, subdict in self.args.items():
            if not type(subdict) is dict:
                setattr(args, category, subdict)
            else:
                for k, v in subdict.items():
                    setattr(args, k, v)
        return args


    def check_run_type(self):
        if self.args['simulation']['fq']:
            self.args['sim_run'] = True
            self.args['live_run'] = False
        elif self.args['live']['device']:
            self.args['live_run'] = True
            self.args['sim_run'] = False
        else:
            raise ValueError("Need either fastq for simulation or device for live run")


def impute_args(args_aeons, args_readfish):
    # check if there are multiple conditions - set split_flowcell
    if type(args_readfish['regions']) is not list:
        raise ValueError("Readfish regions must be specified as array")
    if len(args_readfish['regions']) >= 2:
        args_aeons['live']['split_flowcell'] = True
    else:
        args_aeons['live']['split_flowcell'] = False

    # make sure the names of aeons and regions on flowcell are the same
    for region in args_readfish['regions']:
        if region['name'] in {"control", "Control"}:
            pass
        if region['name'] in {"aeons", "Aeons"}:
            region['name'] = args_aeons['general']['name']


def create_dummy_mmi():
    mm2 = find_exe("minimap2")
    cmd = f'echo " " | {mm2} -x map-ont -d readfish.mmi - '
    execute(cmd)


def validate_readfish_conf(args_rf, prom):
    """
    Minimalist version of readfish validate entry-point
    to validate parsed toml config for readfish

    :param args_rf:
    :param prom:
    :return:
    """
    from readfish._config import Conf
    channels = 3000 if prom else 512
    try:
        conf = Conf.from_dict(args_rf, channels)
    except ValueError("Could not load TOML config for readfish"):
        return 1
    logging.info(conf.describe_experiment())



def set_boss_args(args_rf, name):
    injected = 0
    for region in args_rf['regions']:
        if region['name'] == name:
            region['masks'] = f"out_{name}/masks"
            region['contigs'] = f"out_{name}/contigs"
            injected = 1
    if not injected:
        raise ValueError("BOSS arguments not set, no specified region found")



#%%
def load_config(toml_paths=None):
    # initiate defaults
    conf = Config()
    conf.load_defaults()
    # load command line toml file
    if not toml_paths:
        toml_paths = conf.get_toml_paths()
    args_aeons_file, args_readfish = conf.load_toml_args(toml_paths)
    conf.overwrite_defaults(args_aeons_file)
    # exchange args between the tools
    impute_args(conf.args, args_readfish)
    # set run to either sim or live
    conf.check_run_type()
    # convert aeons args to Namespace
    args_namespace = conf.convert_to_namespace()
    # initialise a log file in the output folder
    init_logger(logfile=f'{args_namespace.name}.aeons.log', args=args_namespace)
    # config settings for readfish
    if args_namespace.live_run:
        # inject arguments for readfish where to look for new masks and contigs
        set_boss_args(args_rf=args_readfish, name=conf.args['general']['name'])
        # validate readfish args
        validate_readfish_conf(args_readfish, prom=conf.args['live']['prom'])
        # create empty mmi for readfish
        create_dummy_mmi()
        # write toml for readfish
        _ = args_readfish.pop('channels', None)
        rtoml.dump(args_readfish, file=Path(f'readfish.toml'), pretty=True)

    return args_namespace


