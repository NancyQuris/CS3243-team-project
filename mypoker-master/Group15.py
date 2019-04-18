import numpy as np
class CardFeatureVectorCompute:
    feature_num = 18

    # trained hand strength
    street_1 = np.array([0.20846469, 0.19166184, 0.20643419, 0.        , 0.        ,
       0.        , 0.        , 0.20643419, 0.18700509, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        ], dtype=np.float64)
    street_2 = np.array([0.084593  , 0.07380559, 0.08398567, 0.08350848, 0.09543826,
       0.        , 0.        , 0.08350848, 0.07690656, 0.07785753,
       0.07089699, 0.        , 0.08180422, 0.        , 0.06362551,
       0.07635061, 0.04771913, 0.        ], dtype=np.float64)
    street_3 = np.array([0.0778725 , 0.06456338, 0.07101972, 0.08495182, 0.093447  ,
       0.        , 0.        , 0.07056202, 0.06819105, 0.0674895 ,
       0.06371386, 0.        , 0.06674786, 0.093447  , 0.05339829,
       0.062298  , 0.062298  , 0.        ], dtype=np.float64)
    street_4 = np.array([0.06753543, 0.05284578, 0.05959009, 0.08011556, 0.0436994 ,
       0.        , 0.        , 0.06117916, 0.05726128, 0.05826586,
       0.05826586, 0.0655491 , 0.04767207, 0.08739879, 0.05669111,
       0.05826586, 0.05826586, 0.08739879], dtype=np.float64)
    streets = [street_1, street_2, street_3, street_4]

    def get_strength(self, street):
        return self.streets[street]

    def get_copy_for_train(self):
        copy = []
        for i in range(len(self.streets)):
            d = dict()
            d['raise'] = np.copy(self.streets[i])
            d['call'] = np.copy(self.streets[i])
            d['fold'] = np.multiply(np.copy([self.streets[i]]), -1)
            copy.append(d)
        return copy

    # card_feature_compute
    def fetch_feature(self, hole, community):
        cards = hole + community
        assert(len(hole) == 2)
        feature_vector = self.hole_eval(hole)
        feature_vector += self.value_count(cards)
        feature_vector += self.suit_count(cards)
        feature_vector += self.seq_count(cards)
        return feature_vector

    def hole_eval(self, cards):
        max_val = max(cards[0].rank, cards[0].rank)
        diff = abs(cards[0].rank - cards[1].rank)
        scaled_max_val = (max_val - 1) / float(14)
        scaled_diff = (12 - diff) / float(13)
        return [scaled_max_val, scaled_diff]

    def value_count(self, cards):
        memo = 0
        pair_memo = 0
        three_memo = 0
        four_memo = 0
        for card in cards:
            mask = 1 << card.rank
            four_memo |= three_memo & mask
            three_memo |= pair_memo & mask
            pair_memo |= memo & mask
            memo |= mask
        pair = 1 if pair_memo != 0 else 0
        pair_num = self.__count_bit(pair_memo) / float(3)
        three = 1 if three_memo != 0 else 0
        three_num = self.__count_bit(three_memo) / float(2)
        four = 1 if four_memo != 0 else 0
        full_house = 1 if pair_memo != three_memo else 0
        return [pair, pair_num, three, three_num, four, full_house]

    def suit_count(self, cards):
        memo = 0
        two_memo = 0
        three_memo = 0
        four_memo = 0
        five_memo = 0
        for card in cards:
            mask = card.suit
            five_memo |= four_memo & mask
            four_memo |= three_memo & mask
            three_memo |= two_memo & mask
            two_memo |= memo & mask
            memo |= mask

        pair = 1 if two_memo != 0 else 0
        pair_num = self.__count_bit(two_memo) / float(3)
        three = 1 if three_memo != 0 else 0
        three_num = self.__count_bit(three_memo) / float(2)
        four = 1 if four_memo != 0 else 0
        five = 1 if five_memo != 0 else 0
        return [pair, pair_num, three, three_num, four, five]

    def seq_count(self, cards):
        memo = 0
        for card in cards:
            mask = 1 << card.rank
            memo |= mask
        s_len = min(5, max(0, self.__count_longest_seq(memo) - 1))
        two = 1 if s_len > 1 else 0
        three = 1 if s_len > 2 else 0
        four = 1 if s_len > 3 else 0
        five = 1 if s_len > 4 else 0
        return [two, three, four, five]


    def __count_bit(self, n):
        count = 0
        while n:
            count += n & 1
            n = n >> 1
        return count

    def __count_longest_seq(self, n):
        count = 0
        while n:
            count = count + 1
            n = n & (n >> 1)
        return count
