from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
# from rt_player import RTPlayer
from rt_player_sec import RTPlayer
import pprint

#TODO:config the config as our wish
rtplayer = RTPlayer()
for i in range(8):
    config = setup_config(max_round=1000, initial_stack=10000, small_blind_amount=10)
    config.register_player(name="f1", algorithm=RaisedPlayer())
    config.register_player(name="FT2", algorithm=rtplayer)

    game_result = start_poker(config, verbose=2)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(game_result)

pp.pprint(rtplayer.step_theta)
