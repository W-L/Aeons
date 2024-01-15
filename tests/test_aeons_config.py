import argparse
import unittest
import sys
import logging

import aeons.aeons_config
import aeons.aeons_core


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])
sys.path.insert(0, "../Aeons")


SIM_TOML = "../Aeons/tests/toml_sim.toml"
LIVE_TOML = "../Aeons/tests/toml_live.toml"
READFISH_TOML = "../Aeons/tests/test_readfish.toml"


class TestConfig(unittest.TestCase):


    def setUp(self):
        self.config = aeons.aeons_config.Config()
        self.config.load_defaults()
        self.toml_paths = self.config.get_toml_paths(
            args=['--toml', LIVE_TOML, '--toml_readfish', READFISH_TOML])
        self.toml_paths_sim = self.config.get_toml_paths(
            args=['--toml', SIM_TOML])


    def test_load_defaults(self):
        # just testing one arbitrary value
        self.assertEqual(self.config.args['filters']['min_seq_len'], 2500)

    def test_get_toml_paths(self):
        self.assertEqual(self.toml_paths.toml, LIVE_TOML)
        self.assertEqual(self.toml_paths.toml_readfish, READFISH_TOML)

    def test_load_toml_args(self):
        args_aeons, args_rf = self.config.load_toml_args(toml_paths=self.toml_paths)
        # check type
        self.assertIsInstance(args_aeons, dict)
        self.assertIsInstance(args_rf, dict)
        # check two arbitrary values
        self.assertEqual(args_aeons['general']['name'], 'aeons')
        self.assertEqual(args_rf['regions'][0]['no_map'], "stop_receiving")

    def test_load_toml_args_sim(self):
        args_aeons, args_rf = self.config.load_toml_args(toml_paths=self.toml_paths_sim)
        # check type
        self.assertIsInstance(args_aeons, dict)
        self.assertIsInstance(args_rf, dict)
        self.assertFalse(args_rf)
        # check two arbitrary values
        self.assertIsInstance(args_aeons['simulation']['fq'], str)

    def test_overwrite_defaults(self):
        args_aeons, args_rf = self.config.load_toml_args(toml_paths=self.toml_paths)
        self.config.overwrite_defaults(args_from_file=args_aeons)
        self.assertEqual(self.config.args['live']['data_wait'], 30)

    def test_convert_to_namespace(self):
        args = self.config.convert_to_namespace()
        self.assertIsInstance(args, argparse.Namespace)
        self.assertEqual(args.name, 'aeons')

    def test_check_run_type(self):
        args_aeons, _ = self.config.load_toml_args(toml_paths=self.toml_paths)
        self.config.overwrite_defaults(args_from_file=args_aeons)
        self.config.check_run_type()
        self.assertTrue(self.config.args['live_run'])
        self.assertFalse(self.config.args['sim_run'])
        # load a simulation toml
        args_aeons, _ = self.config.load_toml_args(toml_paths=self.toml_paths_sim)
        self.config.overwrite_defaults(args_from_file=args_aeons)
        self.config.check_run_type()
        self.assertFalse(self.config.args['live_run'])
        self.assertTrue(self.config.args['sim_run'])

    def test_impute_args(self):
        args_aeons, args_rf = self.config.load_toml_args(toml_paths=self.toml_paths)
        self.config.overwrite_defaults(args_from_file=args_aeons)
        self.config.check_run_type()
        args = self.config.convert_to_namespace()
        aeons.aeons_config.impute_args(args, args_rf)
        self.assertTrue(args.split_flowcell)

    def test_validate_readfish_conf(self):
        args_aeons, args_rf = self.config.load_toml_args(toml_paths=self.toml_paths)
        r = aeons.aeons_config.validate_readfish_conf(args_rf, prom=False)
        self.assertFalse(r)

    def test_load_config(self):
        toml_paths = argparse.Namespace(toml=LIVE_TOML, toml_readfish=READFISH_TOML)
        args = aeons.aeons_config.load_config(toml_paths=toml_paths)
        self.assertIsInstance(args, argparse.Namespace)
        self.assertTrue(args.live_run)
        self.assertEqual(args.data_wait, 30)
        self.assertEqual(args.name, 'aeons')
        self.assertTrue(args.split_flowcell)


if __name__ == '__main__':
    unittest.main()

