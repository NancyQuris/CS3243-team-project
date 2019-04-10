from pypokerengine.api.game import setup_config, start_poker
from randomplayer import RandomPlayer
import pprint

class Simulation():

    def simulate(self, nGames, nRounds, initial_stack, small_blind_amount):
      # since no api is provided for simulate games with known hole_card value, we just simulate nRounds using random player for convenience purpose
      results = []
      config = setup_config(max_round=nRounds, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
      config.register_player(name="f1", algorithm=RandomPlayer())
      config.register_player(name="FT2", algorithm=RandomPlayer())

      for i in range(nGames):
          game_result = start_poker(config, verbose=0)
          results.append(game_result['players'][0]['stack'] - game_result['rule']['initial_stack'])

      print(sum(results)/len(results))
