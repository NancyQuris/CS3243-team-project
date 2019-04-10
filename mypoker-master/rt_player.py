from pypokerengine.players import BasePokerPlayer
from randomplayer import RandomPlayer
import numpy as np
from trained_hand_feature_strength import Trained_hand_feature
from card_feature_vector import CardFeatureVectorCompute
import pypokerengine.utils.card_utils as Card_util
import pprint
import collections


class RTPlayer(BasePokerPlayer):

  def __init__(self):
    super(BasePokerPlayer, self).__init__()
    ## printer used for debug purpose
    self.cfvc = CardFeatureVectorCompute()
    self.pp = pprint.PrettyPrinter(indent=2)

    ## basic records
    self.hole_card = []
    # stack record keep two player's stack from last round
    self.stack_record = [['py1 last stack', 'py1 current stack'], ['py2 last stack', 'py2 current stack']]
    # seat_id records the position of this player in the seat list (0/1)
    self.seat_id = 0
    self.max_round = 0
    self.small_blind_amount = 0
    self.street_map = {'preflop': 0, 'flop': 1, 'river': 2, 'turn': 3}
    self.rev_street_map = {0: 'preflop', 1:'flop', 2: 'river', 3: 'turn'}

    ## params used in training part
    # update at the start of every round
    self.estimated_step_rewards = []
    # alpha is the learning rate used in reinforcement learning part
    self.alpha = 0.01
    # nParams is the num of attributes in a feature vector
    # theta is the weight vector that needs updated
    self.nParams = 6
    #TODO; better way to initialize the theta?
    self.theta = np.array([1, 0, 0, 0, 0, 0])

    # Input:
    #   theta: parameter for current model Qhat
    #   hand: hand number
    #   isMe: boolean, True for the player, False for opponents.
    #   epsilon: chance of making a random move
    # Output:
    #   A tuple of form (isGII, qhat, phi) describing the action
    #   taken, its value, and its feature vector.
  def declare_action(self, valid_actions, hole_card, round_state):
    # check if raise 4 times alr, cannot raise any more
    # flag is true -- still can raise, otherwise cannot
    flag = True
    current_round = round_state['action_histories'][round_state['street']]
    uuid1 = round_state['seats'][0]['uuid']
    uuid2 = round_state['seats'][1]['uuid']
    # raise count for current round
    raiseCount = collections.defaultdict(int)
    for action_details in current_round:
        if action_details['action'] is 'RAISE' or 'BIGBLIND':
            # Big blind is also considered as 'RAISE'
            raiseCount[action_details['uuid']] += 1

    if raiseCount[uuid1] >= 4 or raiseCount[uuid2] >= 4:
        flag = False

    # calculate card strength

    thf = Trained_hand_feature()
    card_feature = self.cfvc.fetch_feature(Card_util.gen_cards(hole_card), Card_util.gen_cards(round_state['community_card']))
    card_strength = np.dot(card_feature, thf.get_strength(self.street_map[round_state['street']]))

    # def act(self, theta, card_strength, isMe, my_stack, opponent_stack, curr_action, epsilon):
    # feature vector for different action
    isMe = True
    my_stack = round_state['seats'][self.seat_id]['stack']
    opponent_stack = round_state['seats'][1-self.seat_id]['stack']

    # get the feature vector for every possible action
    phiRAISE = self.phi(card_strength, isMe, my_stack, opponent_stack, 'raise')
    phiCALL = self.phi(card_strength, isMe, my_stack, opponent_stack, 'call')
    phiFOLD = self.phi(card_strength, isMe, my_stack, opponent_stack, 'fold')

    # value for taking different action
    qRAISE = self.evalModel(self.theta, phiRAISE)
    qCALL = self.evalModel(self.theta, phiRAISE)
    qFOLD = self.evalModel(self.theta, phiFOLD)

    # choose the action with highest value as the next action
    # with same quality, choose in order 'raise' > 'call' > 'fold'
    next_action = ''
    if qRAISE >= np.amax([qCALL, qFOLD]):
        next_action = 'raise'
    elif qCALL >= np.amax([qRAISE, qFOLD]):
        next_action = 'call'
    else:
        next_action = 'fold'

    # actions in valid_actions other than the current 'next_action'
    remain_actions = []
    for act in valid_actions:
        if act['action'] is not next_action:
            remain_actions.append(act['action'])
    assert len(remain_actions) in [1, 2]

    if np.random.rand() < self.epsilon(round_state['round_count'])/2 \
        or (flag is False and next_action is 'raise') \
        or (len(valid_actions) is 2 and next_action is 'raise'):
        # Condition for determining whether the next action should be randomly chosen
        # 1. random_number < epsilon/2
        # 2. next_action is 'raise' but raise num alr 4, cannot raise anymore
        # 3. next_action not in valid_action (there are only two kind of valid_action set: with/withou 'raise')
        # if any of this is satisfied, next_action will be randomly chosen from remain_actions
        if (len(remain_actions) is 1):
            # only remain 1 action, choose this by default
            next_action = remain_actions[0]
        else :
            # remain 2 actions, randomly choose from these two
            if np.random.rand() < 0.5:
                next_action = remain_actions[0]
            else:
                next_action = remain_actions[1]

    # next_action is finalised, store the 'q' and 'phi'
    if next_action is 'raise':
        prob = qRAISE / (qRAISE + qCALL + qFOLD)
        self.estimated_step_rewards.append([next_action, qRAISE, phiRAISE, prob])
    elif next_action is 'call':
        prob = qCALL / (qRAISE + qCALL + qFOLD)
        self.estimated_step_rewards.append([next_action, qCALL, phiCALL, prob])
    else:
        prob = qFOLD / (qRAISE + qCALL + qFOLD)
        self.estimated_step_rewards.append([next_action, qFOLD, phiFOLD, prob])

    # print(next_action)

    return next_action # action returned here is sent to the poker engine

  def receive_game_start_message(self, game_info):
    # initialise stack record when enters the first round
    initial_stack = game_info['rule']['initial_stack']
    self.max_round = game_info['rule']['max_round']
    self.small_blind_amount = game_info['rule']['small_blind_amount']

    if game_info['seats'][0]['uuid'] is self.uuid:
        self.seat_id = 0;
    else:
        self.seat_id = 1;

    assert(self.seat_id in [0,1])

    self.stack_record = [[initial_stack] * 2, [initial_stack] * 2]
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    self.estimated_step_rewards = []
    self.hole_card = hole_card
    pass

  def receive_street_start_message(self, street, round_state):
    self.pp.pprint("street: " + street)
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):

    # start training at the end of every round
    self.stack_record = [[self.stack_record[0][1], round_state['seats'][0]['stack']],
                         [self.stack_record[1][1], round_state['seats'][1]['stack']]]

    curr_lvl_reward = [self.stack_record[0][1] - self.stack_record[0][0],
                       self.stack_record[1][1] - self.stack_record[1][0]]
    # print(curr_lvl_reward)

    oppo_hole_card = []
    # backtrack every step
    self.estimated_step_rewards = self.estimated_step_rewards[::-1]
    # step denotes which step's data is been retrieve
    step = 0
    # round order: 'preflop', 'flop', 'river', 'turn'
    for srt in ['preflop', 'flop', 'river', 'turn']:
        if srt not in round_state['action_histories']:
            continue


        curr_actions = round_state['action_histories'][srt][::-1]
        for act in curr_actions:
            if step is len(self.estimated_step_rewards):
                break

            if self.isMe(act['uuid']):
                action, qACT, phiACT, prob = self.estimated_step_rewards[step]

                # update theta
                self.theta += self.alpha * (curr_lvl_reward[self.seat_id] - qACT) * phiACT
                curr_lvl_reward[self.seat_id] *= prob
                step += 1
            else:
                # learn from opponent's behaviour
                # get the feature vector for every possible action
                # since we cannot know the opponent's card here, card_strength is set to 0
                card_strength = 0
                isMe = False
                my_stack = self.stack_record[self.seat_id][1]
                opponent_stack = self.stack_record[1-self.seat_id][1]
                phiRAISE = self.phi(card_strength, isMe, my_stack, opponent_stack, 'raise')
                phiCALL = self.phi(card_strength, isMe, my_stack, opponent_stack, 'call')
                phiFOLD = self.phi(card_strength, isMe, my_stack, opponent_stack, 'fold')

                # value for taking different action
                qRAISE = self.evalModel(self.theta, phiRAISE)
                qCALL = self.evalModel(self.theta, phiRAISE)
                qFOLD = self.evalModel(self.theta, phiFOLD)

                if act['action'] is 'raise':
                    prob = qRAISE / (qRAISE + qCALL + qFOLD)
                    self.theta += self.alpha * (curr_lvl_reward[1-self.seat_id] - qRAISE) * phiRAISE
                    curr_lvl_reward[1 - self.seat_id] *= prob
                elif act['action'] is 'call':
                    prob = qCALL / (qRAISE + qCALL + qFOLD)
                    self.theta += self.alpha * (curr_lvl_reward[1-self.seat_id] - qCALL) * phiCALL
                    curr_lvl_reward[1 - self.seat_id] *= prob
                else:
                    prob = qFOLD / (qRAISE + qCALL + qFOLD)
                    self.theta += self.alpha * (curr_lvl_reward[1-self.seat_id] - qFOLD) * phiFOLD
                    curr_lvl_reward[1 - self.seat_id] *= prob


  def isMe(self, uuid):
      return uuid is self.uuid

  #
  def sigmoid(self, my_stack, opponent_stack):
      if my_stack > opponent_stack:
          return my_stack/opponent_stack - 1
      else:
          return 1 - my_stack/opponent_stack

    # feature range: 0-1
    # nParams = 6
    # Input:
    # state
    #   (int within range 0-1) card_strength: my estimated hand strength
    #   isMe: boolean true if the player is me
    #   mystack
    #   opponent stack
    # action
    #   action: Fold:0, Call: 0.5, Raise: 1
    #   Explanation: In our strategy, a player tends to raise if he think he can win.
    # Output:
    #   numpy array containing features describing a state and action
  def phi(self, card_strength, isMe, mystack, opponent_stack, curr_action):
    return np.array([1, # for convenience purpose
                     card_strength if curr_action is not 'fold' else 0, # 1. contains 18 feature representations, should normalised to 1,
                     # since the model will learn better if all the features have about the same magnitudes
                     # 2. when the action is fold, the particular holding doesn't have any effect on the result (neglecting minor card removal effects)
                     1 if curr_action is 'raise' else 0.5 if curr_action is 'call' else 0,
                     1 if isMe else -1,
                     1 if isMe and (curr_action is 'raise') else 0,
                     self.sigmoid(mystack, opponent_stack),
                     ])

    # Inputs:
    #   theta: vector of parameters of our model
    #   phi: vector of features
    # Output:
    #   Qhat(phi; theta), an estimate of the action-value
  def evalModel(self, theta, phi):
    return np.dot(theta, phi)

    # Input:
    #   nRound: total number of rounds playing
    #   i: current round
    # Output:
    #   Fraction of the time we should choose our action randomly.
    # This is because we need to take all the actions in all the states at least occasionally if we want to end up with good estimates of each possibility's value
    # To do this, we can have the players act randomly some fraction epsilon of the time, otherwise use their (current-estimated) best options.
    # epsilon will shrink over time
  def epsilon(self, curr_round):
    '''
    base = 0.1
    fraction = (self.max_round - curr_round) / float(self.max_round
    return math.exp(-)
    '''
    return (self.max_round - curr_round) / self.max_round

def setup_ai():
  return RandomPlayer()
