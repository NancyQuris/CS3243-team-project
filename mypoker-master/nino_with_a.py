from pypokerengine.players import BasePokerPlayer
from randomplayer import RandomPlayer
import numpy as np
from trained_hand_feature_strength import Trained_hand_feature
from trained_other_feature_weight import Trained_other_feature
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
        self.nParams = 5
        # self.learn_factor = [0.01, 0.01, 0.01, 0.01]
        self.learn_factor = 0
        self.scale_factor = 0.5

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

        ## params for learning from opponents' behaviour
        self.opp_steps = []
        self.money_record = []

        #update every street
        self.feature_vector = np.ones(self.nParams + 1)
        self.q_suggest = {'raise': 0, 'call': 0, 'fold': 0}
        self.street_idx = 0

        self.call_static_prob = 0.5
        self.raise_static_prob = 0.4
        self.fold_static_prob = 0.1

        #TODO: how to initialize theta
        # Trained_other_feature().get_weights()
        self.step_theta = [self.theta_single_step(self.nParams + 1),\
             self.theta_single_step(self.nParams + 1),\
             self.theta_single_step(self.nParams + 1),\
             self.theta_single_step(self.nParams + 1)]
        # helper to compute the strength
        self.cfvc = CardFeatureVectorCompute()
        self.thf = Trained_hand_feature()
        self.eps = 0
        self.game_count = 0

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
        card_feature = self.cfvc.fetch_feature(Card_util.gen_cards(self.hole_card),
                                               Card_util.gen_cards(round_state['community_card']))
        card_feature = self.get_transferred_vec(card_feature)
        card_strength = np.dot(card_feature, self.thf.get_strength(self.street_idx))
        my_stack = round_state['seats'][my_id]['stack']
        opp_stack = round_state['seats'][opp_id]['stack']
        my_bet = self.stack_record[my_id][1] - my_stack
        opp_bet = self.stack_record[opp_id][1] - opp_stack
        my_total_gain = self.total_gain[my_id]
        self.money_record.append([my_stack, opp_stack, my_bet, opp_bet, self.total_gain])

        # get the feature vector for every possible action
        isMe = True
        feature_vec = self.phi(card_strength, isMe, my_stack, opp_stack, my_bet, opp_bet, my_total_gain)
        self.feature_vector = feature_vec

        # value for taking different action
        q_raise = np.dot(self.step_theta[self.street_idx]['raise'], feature_vec)
        q_call = np.dot(self.step_theta[self.street_idx]['call'], feature_vec)
        q_fold = np.dot(self.step_theta[self.street_idx]['fold'], feature_vec)
        # print('raise %10.6f, call %10.6f, fold %10.6f' % (q_raise, q_call, q_fold))

        self.q_suggest['raise'] = q_raise
        self.q_suggest['call'] = q_call
        self.q_suggest['fold'] = q_fold

        # choose action
        next_action, probability = self.action_select_helper(valid_actions, flag)
        # print('next action: %s' % next_action)
        expected_reward = self.q_suggest[next_action]
        self.estimated_step_rewards.append([next_action, expected_reward, probability, self.street_idx, self.feature_vector])
        print(next_action)
        return next_action  # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        # initialise stack record when enters the first round
        self.initial_stack = game_info['rule']['initial_stack']
        self.max_round = game_info['rule']['max_round']
        self.small_blind_amount = game_info['rule']['small_blind_amount']

        if game_info['seats'][0]['uuid'] is self.uuid:
            self.seat_id = 0
        else:
            self.seat_id = 1

        self.stack_record = [[self.initial_stack] * 2, [self.initial_stack] * 2]

        self.game_count += 1
        # self.learn_factor = 0 if self.game_count > 10 else 0.01
        self.learn_factor = 0.01
        # self.learn_factor = np.floor(10 / self.game_count) / float(100)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.estimated_step_rewards = []
        self.total_gain = [self.stack_record[0][1] - self.initial_stack, self.stack_record[1][1] - self.initial_stack]
        self.bet_has_placed = [0, 0]
        self.hole_card = hole_card

        self.money_record = []

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

        my_id = self.seat_id
        opp_id = 1 - my_id
        true_reward = [self.stack_record[0][1] - self.stack_record[0][0],\
                       self.stack_record[1][1] - self.stack_record[1][0]][my_id]
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
            delta = np.multiply(feature_vec, (true_reward - expected_reward) * prob * self.learn_factor)
            # print('true reward: %6.3f, expected reward: %6.3f' % (true_reward, expected_reward))
            self.step_theta[step_idx][action] = np.add(self.step_theta[step_idx][action], delta)
        # self.pp.pprint(self.step_theta)

        self.accumulate += true_reward
        self.results.append(self.accumulate)

        # learn from opponent's behaviour
        # using the same method to learn
        # backward propagation
        # read feature
        isMe = False
        opp_id = 1 - self.seat_id
        # use hand info to calculate the opponent's hand strength
        opp_hole_card = []
        ####### depends on whether we know oppo hole card or not
        card_feature = self.cfvc.fetch_feature_2(Card_util.gen_cards(round_state['community_card']))
        card_feature = self.get_transferred_vec(card_feature)
        card_strength = np.dot(card_feature, self.thf.get_strength(self.street_idx))

        cum_opp_bet = 0
        cum_my_bet = 0
        my_bet_order = []
        for curr_str_idx in range(4):
            curr_street = self.rev_street_map[curr_str_idx]
            if curr_street not in round_state['action_histories'].keys():
                continue

            # self.pp.pprint(round_state)
            for action in round_state['action_histories'][curr_street]:
                if action['uuid'] is not self.uuid and action['action'] is not 'FOLD':
                    cum_opp_bet += action['amount']
                    self.opp_steps.append([action['action'], curr_str_idx, action['amount']])
                else:
                    # print("NINOMI!!!!!")
                    # self.pp.pprint(action)
                    # record my subsequent bet
                    if action['action'] is not 'FOLD':
                        cum_my_bet += action['amount']
                        my_bet_order.append([action['amount'], curr_str_idx])
                    else:
                        my_bet_order.append([0, curr_str_idx])
                    # TODO: error-prone


        # all actions have been summarised
        # reverse in order for backtrack propagation
        self.opp_steps = self.opp_steps[::-1]
        my_bet_order = my_bet_order[::-1]
        my_bet_idx = 0
        curr_opp_stack = self.stack_record[opp_id][1]
        curr_my_stack = self.stack_record[my_id][1]
        opp_true_reward = [self.stack_record[0][1] - self.stack_record[0][0],\
                           self.stack_record[1][1] - self.stack_record[1][0]][opp_id]
        prob = 1

        for step in self.opp_steps:
            action = step[0].lower()
            curr_str_idx = step[1]
            add_amount = step[2]

            opp_bet = cum_opp_bet
            cum_opp_bet -= add_amount  # the oppo bet used for next action

            my_bet = cum_my_bet
            cum_my_bet -= my_bet_order[my_bet_idx][0]
            my_bet_idx += 1
            if my_bet_idx == len(my_bet_order):
                my_bet_idx -= 1

            # TODO: curr stack could be problematic
            feature_vec = self.phi(card_strength, isMe, curr_opp_stack, curr_my_stack, opp_bet, my_bet, self.total_gain[opp_id])
            # TODO: what is the small blind
            if action == 'bigblind' or 'smallblind':
                action = 'raise'
            q_act = np.dot(self.step_theta[self.street_idx][action], feature_vec)

            curr_opp_stack -= opp_bet
            curr_my_stack -= my_bet

            probability = self.scale_factor
            expected_reward = q_act

            # update the theta
            prob *= probability
            delta = np.multiply(feature_vec, (opp_true_reward - expected_reward) * prob * self.learn_factor)
            # print('true reward: %6.3f, expected reward: %6.3f' % (true_reward, expected_reward))
            self.step_theta[curr_str_idx][action] = np.add(self.step_theta[curr_str_idx][action], delta)
        # self.pp.pprint(self.step_theta)

    def action_select_helper(self, valid_actions, flag):
        valid_acts = list(map(lambda x: x['action'], valid_actions))
        # remove raise if raise is not allowed
        if not flag and 'raise' in valid_acts:
            valid_acts.remove('raise')

        action_to_choose = { x: self.q_suggest[x] for x in valid_acts }
        num_valid = len(valid_acts)
        assert(num_valid > 0)
        max_action = max(action_to_choose, key=action_to_choose.get)

        remains = []
        for act in valid_acts:
            if act != max_action:
                remains.append(act)

        if self.ifRandom:
            action = ''
            if len(remains) is 1:
                action = remains[0]
            else:
                action = remains[0] if np.random.rand() < 0.5 else remains[1]
            return action, self.scale_factor

        return max_action, self.scale_factor

    def theta_single_step(self, length):
        return {'raise': np.zeros(length),
                'call': np.zeros(length),
                'fold': np.zeros(length)}

    def phi(self, hand_strength, isMe, my_stack, opp_stack, my_bet, opp_bet, my_total_gain):
        return np.array([
            1,
            1 if isMe else 0.5,
            hand_strength,
            # self.diff_normal(my_bet, opp_bet),
            (my_bet - opp_bet) / float(10),
            self.diff_normal(my_stack, opp_stack),
            self.diff_normal(my_total_gain, - my_total_gain)
        ])

    def sigmoid(self, x):
        return float(1) / (1 + np.exp(-x))

    def diff_normal(self, x, y):
        if y == 0:
            return 1 if x > 0 else -1
        return self.sigmoid(float(x - y) / np.abs(y))

    def epsilon(self, round_count):
        # self.eps = round_count / self.max_round
        self.eps = float(1) / round_count
        # self.eps = float(1) / ((self.game_count - 1) * self.max_round + round_count)
        # self.eps = 0.1 + 0.9 * float(1) / round_count
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
