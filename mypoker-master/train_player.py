from pypokerengine.players import BasePokerPlayer
from randomplayer import RandomPlayer
from time import sleep
import pprint
import collections
from feature_strength_offline import FeatureStrengthOffline

class TrainedPlayer(BasePokerPlayer):

  def __init__(self, featureStrengthOffline):
    self.pp = pprint.PrettyPrinter(indent=2)
    self.featureStrengthOffline = featureStrengthOffline
    self.hole_card = []

  def declare_action(self, valid_actions, hole_card, round_state):
    #  self.pp.pprint(round_state)
   # if round_state['street'] is 'preflop':
     #   self.pp.pprint("preflop action number: " + len(round_state['street']['action_histories']['preflop']))

    for i in valid_actions:
        if i["action"] == "raise":
            action = i["action"]
            return action  # action returned here is sent to the poker engine
    action = valid_actions[1]["action"]
    return action # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
#    self.pp.pprint(game_info)
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    self.hole_card = hole_card
    pass

  def receive_street_start_message(self, street, round_state):
    community_card = round_state['community_card']
    self.featureStrengthOffline.raw_feed(self.hole_card, community_card, len(community_card))
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    #self.pp.pprint(round_state)
    #self.pp.pprint(winners)
    # self.pp.pprint(hand_info)
    # check result and determine who is the winner
    result = 1 if winners[0]['uuid'] is self.uuid else 0
    self.featureStrengthOffline.feed_result(result)
    pass

def setup_ai():
  return RandomPlayer()
