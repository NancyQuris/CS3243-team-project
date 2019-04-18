from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint

class RandomPlayer(BasePokerPlayer):

  def __init__(self):
    self.pp = pprint.PrettyPrinter(indent=2)

  def declare_action(self, valid_actions, hole_card, round_state):
    #self.pp.pprint(round_state)
    r = rand.random()
    if r <= 0.5:
      call_action_info = valid_actions[1]
    elif r<= 0.9 and len(valid_actions ) == 3:
      call_action_info = valid_actions[2]
    else:
      call_action_info = valid_actions[0]
    action = call_action_info["action"]
    #print(action)
    return action  # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    # self.pp.pprint(game_info)
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    # self.pp.pprint(round_count)
    pass

  def receive_street_start_message(self, street, round_state):
    #self.pp.pprint("1 ----- " + street)
    #self.pp.pprint("2 ----- " + round_state['street'])
    pass

  def receive_game_update_message(self, action, round_state):
   # self.pp.pprint(round_state)
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    print("NINONONONONONONONO")
    self.pp.pprint(hand_info)
    pass

def setup_ai():
  return RandomPlayer()
