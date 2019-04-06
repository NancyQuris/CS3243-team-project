from pypokerengine.players import BasePokerPlayer
from time import sleep
import pprint
import collections

class RaisedPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    # check if raise 4 times alr, cannot raise any more

  # round_state
    pp = pprint.PrettyPrinter(indent=2)

    flag = True
    current_round = round_state['action_histories'][round_state['street']]
    uuid1 = round_state['seats'][0]['uuid']
    uuid2 = round_state['seats'][1]['uuid']

  # (current_round)
    raiseCount = collections.defaultdict(int)

    for action_details in current_round:
        if action_details['action'] is 'RAISE' or 'BIGBLIND':
            # Big blind is also considered as 'RAISE'
            raiseCount[action_details['uuid']] += 1

    if raiseCount[uuid1] >= 4 or raiseCount[uuid2] >= 4:
        flag = False

    pp.pprint("ROUND STATE")
    pp.pprint(round_state)

    for i in valid_actions:
        if i["action"] == "raise":
            action = i["action"]
            return action  # action returned here is sent to the poker engine
    action = valid_actions[1]["action"]
    return action # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return RandomPlayer()
