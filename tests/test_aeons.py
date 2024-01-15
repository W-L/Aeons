import unittest
from pathlib import Path
from argparse import Namespace
import sys
import logging


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])
sys.path.insert(0, "../Aeons")

import aeons.aeons_config
import aeons.aeons_core


SIM_TOML = "../Aeons/tests/toml_sim.toml"
LIVE_TOML = "../Aeons/tests/toml_live.toml"
READFISH_TOML = "../Aeons/tests/test_readfish.toml"


# TODO write actual unit tests
#
#
# class TestSim(unittest.TestCase):
#
#     def setUp(self):
#         pass
#
#     def test_full(self):
#         tomls = Namespace(toml=SIM_TOML, toml_readfish=None)
#         self.args = aeons.aeons_config.load_config(toml_paths=tomls)
#         self.ar = aeons.aeons_core.AeonsRun(args=self.args)
#         self.ar.process_batch_sim()
#         # check outcome
#         out_dir = self.ar.out_dir
#         cpath = f'{out_dir}/masks/aeons.npz'
#         markerfile_masks = f'{out_dir}/masks/masks.updated'
#         mmi_path = f'{out_dir}/contigs/aeons.mmi'
#         markerfile_contigs = f'{out_dir}/contigs/contigs.updated'
#         self.assertTrue(Path(cpath).is_file())
#         self.assertTrue(Path(markerfile_masks).is_file())
#         self.assertTrue(Path(mmi_path).is_file())
#         self.assertTrue(Path(markerfile_contigs).is_file())




# TODO write tests for live version while playback is running
#
# class TestLive(unittest.TestCase):
#
#     def setUp(self):
#         pass
#
#
#     def test_live_config(self):
#         tomls = Namespace(toml=LIVE_TOML, toml_readfish=READFISH_TOML)
#         self.args = aeons.aeons_config.load_config(toml_paths=tomls)
#         self.ar = aeons.aeons_core.AeonsRun(args=self.args)




if __name__ == '__main__':
    unittest.main()