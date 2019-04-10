class CardFeatureCompute:
    feature_num = 18

    def fetch_feature(self, hole, community):
        print(hole)
        cards = hole + community
        assert(len(hole) == 2)
        feature_dict = self.hole_eval(cards)
        feature_dict.update(self.value_count(cards))
        feature_dict.update(self.suit_count(cards))
        feature_dict.update(self.seq_count(cards))
        return feature_dict

    def hole_eval(self, cards):
        max_val = max(cards[0].rank, cards[0].rank)
        diff = abs(cards[0].rank - cards[1].rank)
        scaled_max_val = (max_val - 1) / float(14)
        scaled_diff = (12 - diff) / float(13)
        return {
            'HIGH_CARD': scaled_max_val,
            'DIFF': scaled_diff
        }

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
        return {
            "PAIR_VAL": pair,
            "PAIR_VAL_NUM": pair_num,
            "THREE_SAME_VAL": three,
            "THREE_SAME_VAL_NUM": three_num,
            "FOUR_SAME_VAL": four,
            "FULL_HOUSE": full_house
        }

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
        return {
            "PAIR_SUIT": pair,
            "PAIR_SUIT_NUM": pair_num,
            "THREE_SUIT": three,
            "THREE_SUIT_NUM": three_num,
            "FOUR_SUIT": four,
            "FIVE_SUIT": five
        }

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
        return {
            "TWO_SEQ": two,
            "THREE_SEQ": three,
            "FOUR_SEQ": four,
            "FIVE_SEQ": five
        }


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
