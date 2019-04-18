from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from call_player import CallPlayer
# from rt_player import RTPlayer
# from rt_player_sec import RTPlayer
# from rt_player_thr import RTPlayer
# from Ninomiya import RTPlayer
from nino_with_a import RTPlayer
import pprint
import matplotlib.pyplot as plt

#TODO:config the config as our wish
rtplayer = RTPlayer()
for i in range(1):
    config = setup_config(max_round=1000, initial_stack=10000, small_blind_amount=10)
    config.register_player(name="f1", algorithm=CallPlayer())
    config.register_player(name="FT2", algorithm=rtplayer)

    game_result = start_poker(config, verbose=0)
    pp = pprint.PrettyPrinter(indent=2)

    pp.pprint(game_result)
    pp.pprint(rtplayer.step_theta)


y_val_list = rtplayer.get_result()
print(len(y_val_list))
x_val_list = list(range(1, len(y_val_list) + 1))
print(len(x_val_list))
plt.scatter(x_val_list, y_val_list, s=10)

plt.title("Result ")
plt.xlabel("Number")
plt.ylabel("Average Reward")
plt.show()

