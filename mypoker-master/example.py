from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from train_player import TrainedPlayer
from feature_strength_offline import FeatureStrengthOffline
import pprint

#TODO:config the config as our wish
config = setup_config(max_round=100, initial_stack=10000, small_blind_amount=10)

feature_strength_train = FeatureStrengthOffline()

config.register_player(name="f1", algorithm=RandomPlayer())
config.register_player(name="FT2", algorithm=RandomPlayer())

game_result = start_poker(config, verbose=0)
pp = pprint.PrettyPrinter(indent=2)
pp.pprint("------------------GAME RESULT--------------------------")
pp.pprint(game_result)
feature_strength_train.feed_self_feature_prob_map()
pp.pprint(feature_strength_train.output_feature_map())
pp.pprint(feature_strength_train.output_weight_suggest())
