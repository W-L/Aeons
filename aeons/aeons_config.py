import argparse
from argparse import Namespace
from pathlib import Path
import logging
import sys
from typing import Tuple, Dict

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
        """
        Initialise the configuration for AEONS,
        by loading the path to the defaults toml
        """
        self.template_toml = Path(sys.argv[0]).parent / "aeons_template.toml"
        if not self.template_toml.is_file():
            raise FileNotFoundError("Template TOML with configuration defaults not found. Exiting")


    def load_defaults(self):
        """
        Load the TOML containing defaults

        :return:
        """
        self.args = rtoml.load(self.template_toml)


    @staticmethod
    def get_toml_paths() -> argparse.Namespace:
        """
        Parse TOML paths given as arguments on the command line

        :return: argparse.Namespace with paths to the two TOMLs
        """
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
    def load_toml_args(toml_paths: argparse.Namespace) -> Tuple[Dict, Dict]:
        """
        Load the TOML files into dictionaries using rTOML

        :param toml_paths: Paths to TOMLs as parsed arguments
        :return: Tuple with parsed TOML dictionaries
        """
        args_aeons_file = rtoml.loads(Path(toml_paths.toml).read_text(encoding="utf-8"))
        if toml_paths.toml_readfish:
            args_readfish_file = rtoml.loads(Path(toml_paths.toml_readfish).read_text(encoding="utf-8"))
        else:
            args_readfish_file = dict()
        return args_aeons_file, args_readfish_file


    def overwrite_defaults(self, args_from_file: Dict) -> None:
        """
        Use TOML given on CL to overwrite defaults

        :param args_from_file: parsed contents of given TOML
        :return:
        """
        for category in args_from_file:
            for k, v in args_from_file[category].items():
                self.args[category][k] = v


    def convert_to_namespace(self) -> argparse.Namespace:
        """
        Convert arguments from a parsed Dict to a Namespace object
        For method-style access to attributes

        :return: arguments as a Namespace object
        """
        args = Namespace()
        for category, subdict in self.args.items():
            if not type(subdict) is dict:
                setattr(args, category, subdict)
            else:
                for k, v in subdict.items():
                    setattr(args, k, v)
        return args


    def check_run_type(self) -> None:
        """
        Check if we are running a simulation or a real experiment
        If "fq" under the simulation header is given, we simulate

        :return:
        """
        if self.args['simulation']['fq']:
            self.args['sim_run'] = True
            self.args['live_run'] = False
        elif self.args['live']['device']:
            self.args['live_run'] = True
            self.args['sim_run'] = False
        else:
            raise ValueError("Need either fastq for simulation or device for live run")


def impute_args(args_aeons: Dict, args_readfish: Dict) -> None:
    """
    We set a few arguments in readfish, depending on the config given to AEONS
    And vice-versa some arguments in AEONS are set depending on the readfish config

    :param args_aeons: Config dictionary for AEONS
    :param args_readfish: Config dictionary for readfish
    :return:
    """
    # check if there are multiple conditions in readfish
    # this sets split_flowcell in AEONS
    # alternative is to run only AEONS across an entire flowcell
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


def create_dummy_mmi() -> None:
    """
    To initialise a mappy-mapper, we need some .mmi
    Since we don't have contigs in the beginning we create an empty index

    :return:
    """
    mm2 = find_exe("minimap2")
    cmd = f'echo " " | {mm2} -x map-ont -d readfish.mmi - '
    execute(cmd)


def validate_readfish_conf(args_rf: Dict, prom: bool) -> int:
    """
    Minimalist version of readfish validate entry-point
    to validate parsed toml config for readfish

    :param args_rf: Dict of arguments parsed for readfish
    :param prom: Boolean whether we use a PromethION flowcell
    :return: indicator
    """
    from readfish._config import Conf
    channels = 3000 if prom else 512
    try:
        conf = Conf.from_dict(args_rf, channels)
    except ValueError("Could not load TOML config for readfish"):
        return 1
    logging.info(conf.describe_experiment())
    return 0


def set_boss_args(args_rf: Dict, name: str) -> None:
    """
    Set some arguments in the readfish region that uses AEONS
    I.e. where to look for new masks and indices

    :param args_rf: Dictionary of readfish config
    :param name: experiment name
    :return:
    """
    injected = 0
    for region in args_rf['regions']:
        if region['name'] == name:
            region['masks'] = f"out_{name}/masks"
            region['contigs'] = f"out_{name}/contigs"
            injected = 1
    if not injected:
        raise ValueError("BOSS arguments not set, no specified region found")



#%%
def load_config(toml_paths: argparse.Namespace = None) -> argparse.Namespace:
    """
    Wrapper to load the configurations of AEONS and readfish
    Use readfishes TOML as required, and an additional one for AEONS
    When debugging, paths to the TOMLs can be given as a Namespace
    For simulations, providing a readfish TOML is not necessary

    :param toml_paths: Optional paths to TOML files for debugging (without CL)
    :return: Object containing the config for AEONS
    """
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


