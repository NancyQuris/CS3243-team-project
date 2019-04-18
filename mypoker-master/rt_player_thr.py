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
        self.pp = pprint.PrettyPrinter(indent=2)

        ## basic records
        self.street_map = {'preflop': 0, 'flop': 1, 'river': 2, 'turn': 3}
        self.rev_street_map = {0: 'preflop', 1: 'flop', 2: 'river', 3: 'turn'}
        self.nParams = 21
        self.learn_factor = [0.02, 0.02, 0.02, 0.02]

        ## update every game
        self.initial_stack = 0
        self.seat_id = 0
        self.max_round = 0
        self.small_blind_amount = 0

        ## params used in training part
        ## update at the start of every round
        self.hole_card = []
        self.stack_record = [[0] * 2] * 2
        self.total_gain = [0, 0]
        self.bet_has_placed = [0, 0]
        self.estimated_step_rewards = []
        self.ifRandom = False

        #update every street
        self.feature_vector = np.ones(self.nParams + 1)
        self.q_suggest = {'raise': 0, 'call': 0, 'fold': 0}
        self.street_idx = 0;

        #TODO: how to initialize theta
        self.step_theta = [self.theta_single_step(self.nParams),\
                           self.theta_single_step(self.nParams),\
                           self.theta_single_step(self.nParams),\
                           self.theta_single_step(self.nParams)]
        # helper to compute the strength
        self.cfvc = CardFeatureVectorCompute()
        self.thf = Trained_hand_feature()
        self.eps = 0
        self.game_count = 0

        ## a list to keep record of all results
        self.results = []


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

        # read feature
        my_id = self.seat_id
        opp_id = 1 - my_id
        card_feature = self.cfvc.fetch_feature(Card_util.gen_cards(self.hole_card),
                                               Card_util.gen_cards(round_state['community_card']))
        # card_strength = np.dot(card_feature, self.thf.get_strength(self.street_idx))
        my_stack = round_state['seats'][my_id]['stack']
        opp_stack = round_state['seats'][opp_id]['stack']
        my_bet = self.stack_record[my_id][1] - my_stack
        opp_bet = self.stack_record[opp_id][1] - opp_stack
        my_total_gain = self.total_gain[my_id]

        # get the feature vector for every possible action
        # feature_vec = self.phi(card_strength, my_stack, opp_stack, my_bet, opp_bet, my_total_gain)
        feature_vec = self.phi(card_feature, my_stack, opp_stack, my_bet, opp_bet, my_total_gain)
        self.feature_vector = feature_vec

        # value for taking different action
        q_raise = np.dot(self.step_theta[self.street_idx]['raise'], feature_vec)
        q_call = np.dot(self.step_theta[self.street_idx]['call'], feature_vec)
        q_fold = np.dot(self.step_theta[self.street_idx]['fold'], feature_vec)
        # print('raise %10.6f, call %10.6f, fold %10.6f' % (q_raise, q_call, q_fold))

        self.q_suggest['raise'] = q_raise
        self.q_suggest['call'] = q_call
        self.q_suggest['fold'] = q_fold

        #choose action
        next_action, probability = self.action_select_helper(valid_actions, flag)
        # print('next action: %s' % next_action)
        expected_reward = self.q_suggest[next_action]
        self.estimated_step_rewards.append([next_action, expected_reward, probability, self.street_idx, self.feature_vector])
        return next_action # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        # initialise stack record when enters the first round
        self.initial_stack = game_info['rule']['initial_stack']
        self.max_round = game_info['rule']['max_round']
        self.small_blind_amount = game_info['rule']['small_blind_amount']

        if game_info['seats'][0]['uuid'] is self.uuid:
            self.seat_id = 0;
        else:
            self.seat_id = 1;

        self.stack_record = [[self.initial_stack] * 2, [self.initial_stack] * 2]

        self.game_count += 1

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.estimated_step_rewards = []
        self.total_gain = [self.stack_record[0][1] - self.initial_stack, self.stack_record[1][1] - self.initial_stack]
        self.bet_has_placed = [0, 0]
        self.hole_card = hole_card

        r = np.random.rand()
        self.eps = self.epsilon(round_count)
        if r < self.eps:
            self.ifRandom = True
        else:
            self.ifRandom = False

    def receive_street_start_message(self, street, round_state):
        current_street = self.street_map[round_state['street']]
        self.street_idx = current_street


    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # start training at the end of every round
        self.stack_record = [[self.stack_record[0][1], round_state['seats'][0]['stack']], \
                             [self.stack_record[1][1], round_state['seats'][1]['stack']]]

        true_reward = [self.stack_record[0][1] - self.stack_record[0][0], \
                       self.stack_record[1][1] - self.stack_record[1][0]][self.seat_id]

        # backtrack every step
        self.estimated_step_rewards = self.estimated_step_rewards[::-1]
        prob = 1

        for record in self.estimated_step_rewards:
            action = record[0]
            expected_reward = record[1]
            probability = record[2]
            step_idx = record[3]
            feature_vec = record[4]
            # print('action takes: %s, probability: %6.4f' % (action, prob))

            # update the theta
            prob *= probability
            delta = np.multiply(feature_vec, (true_reward - expected_reward) * prob * self.learn_factor[step_idx])
            # print('true reward: %6.3f, expected reward: %6.3f' % (true_reward, expected_reward))
            self.step_theta[step_idx][action] = np.add(self.step_theta[step_idx][action], delta)
        # self.pp.pprint(self.step_theta)

        self.results.append(true_reward)


    def action_select_helper(self, valid_actions, flag):
        valid_acts = list(map(lambda x: x['action'], valid_actions))
        #remove raise if raise is not allowed
        if not flag and 'raise' in valid_acts:
            valid_acts.remove('raise')

        action_to_choose = { x: self.q_suggest[x] for x in valid_acts }
        num_valid = len(valid_acts)
        assert(num_valid > 0)
        max_action = max(action_to_choose, key=action_to_choose.get)

        if self.ifRandom:
            r = np.random.rand()
            action = ''
            if r < 0.5:
                action = 'call'
            elif r < 0.9 and num_valid == 3:
                action = 'raise'
            else:
                # print('here')
                action = 'fold'
            return action, 0.5
        else:
            return max_action, 0.5

    def theta_single_step(self, length):
        return {'raise' : np.ones(length),
                'call' : np.ones(length),
                'fold' : np.ones(length)}

    def phi(self, card_feature, my_stack, opp_stack, my_bet, opp_bet, my_total_gain):
        #return np.array([
        #   hand_strength,
        #   self.diff_normal(my_bet, opp_bet),
        #   self.diff_normal(my_stack, opp_stack),
        #   self.diff_normal(my_total_gain, - my_total_gain)
        #])
        features = self.get_transferred_vec(card_feature)
        appended = np.array([self.diff_normal(my_bet, opp_bet),
                             self.diff_normal(my_stack, opp_stack),
                             self.diff_normal(my_total_gain, - my_total_gain)])
        return np.append(features, appended)

    def sigmoid(self, x):
        return float(1) / (1 + np.exp(-x))

    def diff_normal(self, x, y):
        if y == 0:
            return 1 if x > 0 else -1
        return self.sigmoid(float(x - y) / np.abs(y))

    def epsilon(self, round_count):
        # self.eps = float(1) / ((self.game_count - 1) * self.max_round + round_count)
        self.eps = float(1) / round_count
        return self.eps

    def get_result(self):
        #i = 0
        #avg = []
        #while i <= len(self.results) - part_size:
        #   sum = 0
        #   for j in range(part_size):
        #       sum += self.results[i + j]
        #   avg.append(float(sum) / part_size)
        #   i += part_size
        #return avg
        return self.results

    def get_transferred_vec(self, vec):
        vfunc_f = np.vectorize(self.zero_to_minus_one)
        return vfunc_f(vec)

    @staticmethod
    def zero_to_minus_one(a):
        return -1 if a < 0.5 else 1

def setup_ai():
    return RandomPlayer()