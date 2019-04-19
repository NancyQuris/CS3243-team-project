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
        self.street_map = {'preflop': 0, 'flop': 1, 'river': 2, 'turn': 3, 'showdown': 4}
        self.rev_street_map = {0: 'preflop', 1: 'flop', 2: 'river', 3: 'turn', 'showdown': 4}
        self.nParams = 8
        self.learn_factor = 0.01
        self.opp_factor = 0.2
        self.scale_factor = 0.5

        ## update every game
        self.initial_stack = 0
        self.seat_id = 0
        self.max_round = 0
        self.small_blind_amount = 0
        self.game_count = 0

        ## params used in training part
        ## update at the start of every round
        self.hole_card = []
        self.community = []
        self.stack_record = [[0] * 2] * 2
        self.total_gain = [0, 0]
        self.estimated_step_rewards = []
        self.ifRandom = False
        self.last_game_result = 0

        ## params for learning from opponents' behaviour
        self.opp_steps = []
        self.money_record = []

        #update every street
        self.feature_vector = np.ones(self.nParams + 1)
        self.q_suggest = {'raise': 0, 'call': 0, 'fold': 0}
        self.street_idx = 0

        #TODO: how to initialize theta
        # Trained_other_feature().get_weights()
        # helper to compute the strength
        self.cfvc = CardFeatureVectorCompute()
        self.thf = Trained_hand_feature()
        self.eps = 0
        self.game_count = 0
        self.step_theta = self.thf.get_strength_essential_initial()

        ## a list to keep record of all results
        self.accumulate = 0
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
        card_feature = self.cfvc.fetch_feature_essential(Card_util.gen_cards(self.hole_card),
                                                         Card_util.gen_cards(round_state['community_card']))
        my_stack = round_state['seats'][my_id]['stack']
        opp_stack = round_state['seats'][opp_id]['stack']
        my_bet = self.stack_record[my_id][1] - my_stack
        opp_bet = self.stack_record[opp_id][1] - opp_stack
        my_total_gain = self.total_gain[my_id]
        self.money_record.append([my_stack, opp_stack, my_bet, opp_bet, self.total_gain])

        # get the feature vector for every possible action
        feature_vec = self.phi(card_feature, my_stack, opp_stack, my_bet, opp_bet, my_total_gain)
        self.feature_vector = feature_vec

        # value for taking different action
        q_raise = np.dot(self.step_theta[self.street_idx]['raise'], feature_vec)
        q_call = np.dot(self.step_theta[self.street_idx]['call'], feature_vec)
        q_fold = np.dot(self.step_theta[self.street_idx]['fold'], feature_vec)

        self.q_suggest['raise'] = q_raise
        self.q_suggest['call'] = q_call
        self.q_suggest['fold'] = q_fold

        # choose action
        next_action, probability = self.action_select_helper(valid_actions, flag)
        expected_reward = self.q_suggest[next_action]
        self.estimated_step_rewards.append([next_action, expected_reward, probability, self.street_idx, self.feature_vector])
        return next_action  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        # initialise stack record when enters the first round
        self.initial_stack = game_info['rule']['initial_stack']
        self.max_round = game_info['rule']['max_round']
        self.small_blind_amount = game_info['rule']['small_blind_amount']
        self.game_count = self.game_count + 1

        if game_info['seats'][0]['uuid'] == self.uuid:
            self.seat_id = 0
        else:
            self.seat_id = 1

        self.stack_record = [[self.initial_stack] * 2, [self.initial_stack] * 2]

        self.game_count += 1
        # self.learn_factor = 0 if self.game_count > 10 else 0.01
        # self.learn_factor = 0.01
        # self.learn_factor = np.floor(10 / self.game_count) / float(100)
        if self.last_game_result >= 1.4 * self.initial_stack:
            self.learn_factor = 0
        elif self.last_game_result >= 1.2 * self.initial_stack:
            self.learn_factor = 0.005
        else:
            self.learn_factor = 0.01

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.estimated_step_rewards = []
        self.total_gain = [self.stack_record[0][1] - self.initial_stack, self.stack_record[1][1] - self.initial_stack]
        self.hole_card = hole_card
        self.community = []
        self.money_record = []

        r = np.random.rand()
        self.eps = self.epsilon(round_count)
        if r < self.eps:
            self.ifRandom = True
        else:
            self.ifRandom = False

    def receive_street_start_message(self, street, round_state):
        current_street = self.street_map[round_state['street']]
        self.community.append(round_state['community_card'])
        self.street_idx = current_street

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        # start training at the end of every round
        my_id = self.seat_id
        opp_id = 1 - my_id

        self.stack_record = [[self.stack_record[0][1], round_state['seats'][0]['stack']], \
                             [self.stack_record[1][1], round_state['seats'][1]['stack']]]
        true_reward = [self.stack_record[0][1] - self.stack_record[0][0],\
                       self.stack_record[1][1] - self.stack_record[1][0]]
        my_last_stack = self.stack_record[my_id][0]
        opp_last_stack = self.stack_record[opp_id][0]

        # backtrack every step for myself
        self.estimated_step_rewards = self.estimated_step_rewards[::-1]
        prob = 1

        for record in self.estimated_step_rewards:
            action = record[0]
            expected_reward = record[1]
            probability = record[2]
            step_idx = record[3]
            feature_vec = record[4]

            # update the theta
            prob *= probability
            delta = np.multiply(feature_vec, (true_reward[my_id] - expected_reward) * prob * self.learn_factor)
            self.step_theta[step_idx][action] = np.add(self.step_theta[step_idx][action], delta)

        self.accumulate += true_reward[my_id]
        self.results.append(self.accumulate)

        # learn from opponent's behaviour if there is a showdown
        if len(hand_info) != 0:
            opp_estimated_step_rewards = []
            opp_hole_card = hand_info[opp_id]['hand']['card']
            opp_uuid = hand_info[opp_id]['uuid']

            bet_accumulate = 0
            my_bet_accumulate = 0
            for curr_str_idx in range(4):
                curr_street = self.rev_street_map[curr_str_idx]
                if curr_street not in round_state['action_histories'].keys():
                    continue

                street_bet = 0
                my_street_bet = 0
                opp_street_community_card = self.community[curr_str_idx]
                for step in round_state['action_histories'][curr_street]:
                    if step['uuid'] == opp_uuid:
                        act = step['action'].lower()
                        street_bet = step['amount']
                        if act != 'fold':
                            street_bet = step['amount']

                        if act in {'call', 'raise', 'fold'}:
                            opp_card_feature = self.cfvc.fetch_feature_essential(Card_util.gen_cards(opp_hole_card),
                                                                       Card_util.gen_cards(opp_street_community_card))
                            opp_stack = opp_last_stack - bet_accumulate
                            my_stack = my_last_stack - my_bet_accumulate
                            opp_total_gain = self.total_gain[opp_id]
                            opp_feature_vec = self.phi(opp_card_feature, opp_stack, my_stack, \
                                                       bet_accumulate + street_bet, my_bet_accumulate + my_street_bet, \
                                                       opp_total_gain)
                            opp_expected_reward = np.dot(self.step_theta[curr_str_idx][act], opp_feature_vec)
                            opp_estimated_step_rewards.append([act, opp_expected_reward, \
                                                               self.scale_factor, curr_str_idx, opp_feature_vec])
                    else:
                        if step['action'].lower != 'fold':
                            my_street_bet = step['amount']

                bet_accumulate += street_bet
                my_bet_accumulate += my_street_bet


            prob = 1
            opp_records = opp_estimated_step_rewards[::-1]
            for record in opp_records:
                action = record[0]
                expected_reward = record[1]
                probability = record[2]
                step_idx = record[3]
                feature_vec = record[4]

                prob *= probability
                delta = np.multiply(feature_vec, (true_reward[opp_id] - expected_reward) * prob * self.learn_factor * self.opp_factor)
                self.step_theta[step_idx][action] = np.add(self.step_theta[step_idx][action], delta)

        self.last_game_result = self.stack_record[self.seat_id][1]



    def action_select_helper(self, valid_actions, flag):
        valid_acts = list(map(lambda x: x['action'], valid_actions))
        # remove raise if raise is not allowed
        if not flag and 'raise' in valid_acts:
            valid_acts.remove('raise')

        action_to_choose = { x: self.q_suggest[x] for x in valid_acts }
        num_valid = len(valid_acts)
        assert(num_valid > 0)
        max_action = max(action_to_choose, key=action_to_choose.get)


        if self.ifRandom:
            action = ''
            if len(valid_acts) == 1:
                action = valid_acts[0]
            elif len(valid_acts) == 2:
                r = np.random.rand()
                action = 'call' if r < 0.8 else 'fold'
            else:
                r = np.random.rand()
                if r < 0.5:
                    action = 'call'
                elif r < 0.9:
                    action = 'raise'
                else:
                    action = 'fold'
            return action, self.scale_factor

        return max_action, self.scale_factor

    def theta_single_step(self, length):
        return {'raise': np.ones(length),
                'call': np.ones(length),
                'fold': np.zeros(length)}

    def phi(self, card_feature, my_stack, opp_stack, my_bet, opp_bet, my_total_gain):
        # combine_other = [
        #    (my_bet - opp_bet) / float(10),
        #   self.diff_normal(my_stack, opp_stack),
        #   self.diff_normal(my_total_gain, - my_total_gain)
        # ]
        combine_other = []
        combined = [1] + card_feature + combine_other
        return np.array(combined)

    def sigmoid(self, x):
        return float(1) / (1 + np.exp(-x))

    def diff_normal(self, x, y):
        if y == 0:
            return 1 if x > 0 else -1
        return self.sigmoid(float(x - y) / np.abs(y))

    def epsilon(self, round_count):
        # self.eps = round_count / self.max_round
        # self.eps = float(1) / round_count
        self.eps = float(1) / ((self.game_count - 1) * self.max_round + round_count)
        # self.eps = 0.1 + 0.9 * float(1) / round_count
        return self.eps

    def get_result(self):
        return self.results

    def get_transferred_vec(self, vec):
        vfunc_f = np.vectorize(self.zero_to_minus_one)
        return vfunc_f(vec)

    @staticmethod
    def zero_to_minus_one(a):
        return -1 if a < 0.5 else 1


def setup_ai():
    return RandomPlayer()
