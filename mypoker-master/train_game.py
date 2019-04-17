from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
# from rt_player import RTPlayer
from rt_player_sec import Group15Player as p2
from rt_player_thr import Group15Player as p3
import pprint
import matplotlib.pyplot as plt

#TODO:config the config as our wish
rt2 = p2()
rt3 = p3()
for i in range(1):
    config = setup_config(max_round=500, initial_stack=10000, small_blind_amount=10)
    config.register_player(name="f1", algorithm=RaisedPlayer())
    config.register_player(name="FT2", algorithm=rt2)

    game_result = start_poker(config, verbose=0)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(game_result)

pp.pprint(rt3.step_theta)
