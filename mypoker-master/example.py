from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from train_player import TrainedPlayer
from feature_strength_offline import FeatureStrengthOffline
import pprint

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=10)

feature_strength_train = FeatureStrengthOffline()

config.register_player(name="f1", algorithm=RandomPlayer())
config.register_player(name="FT2", algorithm=TrainedPlayer(feature_strength_train))

game_result = start_poker(config, verbose=1)
pp = pprint.PrettyPrinter(indent=2)
pp.pprint(game_result)
