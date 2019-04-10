class ComputeHandStrength:

    # function that evaluate hand can be found from pypokerengine<engine<hand_evaluator
    # priority: highcard, pair, twopair, three of a kind, straight, flush, full house, four
    # of a kind, straight flush, royal flush

    @staticmethod
    def normalise_card(card):
        for c in card:
            if "A" in c:
                c = c[0] + "14"
                continue
            if "T" in c:
                c = c[0] + "10"
                continue
            if "J" in c:
                c = c[0] + "11"
                continue
            if "Q" in c:
                c = c[0] + "12"
                continue
            if "K" in c:
                c = c[0] + "13"
                continue

        return card

    #priority: C < D < H < S
    @staticmethod
    def card_color(card):
        card_colour = {"C": 0, "D": 0, "H": 0, "S": 0}
        for c in card:
            colour = c[0]
            if colour == "C":
                card_colour["C"] += 1
            if colour == "D":
                card_colour["D"] += 1
            if colour == "H":
                card_colour["H"] += 1
            if colour == "S":
                card_colour["S"] += 1
        return card_colour

    # call the function with normalised card, return card dictionary
    @staticmethod
    def card_component(nhole_card, ncom_card):
        card = {}
        for i in range(2, 15):
            card[i] = 0

        total_card = nhole_card + ncom_card
        for c in total_card:
            c = int(c[1:])
            card[c] += 1

        return card

    # call the function with normalised card
    @staticmethod
    def check_high_card_value(nhole_card):
        card_0 = nhole_card[0]
        card_1 = nhole_card[1]
        card_0_value = int(card_0[1:])
        card_1_value = int(card_1[1:])
        return max(card_0_value, card_1_value)

    @staticmethod
    def check_if_one_pair(card_dic):
        result = False
        pair_value = 1

        for i in (2, 15):
            if card_dic[i] >= 2:
                result = False
                pair_value = i

        return result, pair_value

    @staticmethod
    def check_if_two_pairs(card_dic):
        result = False
        pair_value = (1, 1)
        has_one_pair, pair_value1 = ComputeHandStrength.check_if_one_pair(card_dic)

        if has_one_pair:
            pair_value2 = 1
            for i in (2, pair_value1 + 1):
                if card_dic[i] >= 2:
                    result = True
                    pair_value2 = i

            pair_value = (pair_value2, pair_value1)

        return result, pair_value

    @staticmethod
    def check_if_three_of_a_kind(card_dic):
        result = False
        value = 1

        for i in (2, 15):
            if (card_dic[i]) >= 3:
                result = True
                value = i
        return result, value

    @staticmethod
    def check_if_four_of_a_kind(card_dic):
        result = False
        value = 1

        for i in (2, 15):
            if (card_dic[i]) >= 4:
                result = True
                value = i

        return result, value

    @staticmethod
    def check_if_straight(card_dic):
        result = False
        value = 1
        evaluate_current_sequence = False

        for i in (2, 11):
            index = range(i, i + 5)
            for k in (0, 5):
                current_index = index[k]
                if card_dic[current_index] > 0:
                    evaluate_current_sequence = True
                    continue
                else:
                    evaluate_current_sequence = False
                    break
            if evaluate_current_sequence:
                result = True
                value = i

        return result, value

    @staticmethod
    def check_if_flush(card_colour):
        result = False
        colour = "NA"

        colour_arr = ["C", "D", "H", "S"]

        for i in (0, 4):
            current_colour = colour_arr[i]
            if card_colour[current_colour] >= 5:
                result = True
                colour = current_colour

        return result, colour

    # what if all five card are the same, cannot detect
    @staticmethod
    def check_if_fullhouse(card_dic):

        has_three_of_kind, value_of_three = ComputeHandStrength.check_if_three_of_a_kind(card_dic)
        has_pair, value_of_pair = ComputeHandStrength.check_if_one_pair(card_dic)

        if has_three_of_kind & has_pair:
            if value_of_three == value_of_pair:
                has_pair = False
                for i in (2, value_of_three):
                    if card_dic[i] >= 2:
                        has_pair = True
                        value_of_pair = i

        return has_three_of_kind & has_pair, value_of_three, value_of_pair


    @staticmethod
    def check_if_straight_flush(nhole_card, ncom_card):
        is_straight_flush = False
        card = nhole_card + ncom_card
        card_dic = ComputeHandStrength.card_component(nhole_card, ncom_card)
        card_colour = ComputeHandStrength.card_color(card)
        is_straight, value = ComputeHandStrength.check_if_straight(card_dic)
        is_flush, colour = ComputeHandStrength.check_if_flush(card_colour)

        if is_straight & is_flush:
            is_straight_flush = True
            for i in (value, value + 5):
                current_card = colour + str(i)
                if current_card in card:
                    continue
                else:
                    is_straight_flush = False
                    break

        return is_straight_flush, value, colour

    @staticmethod
    def check_if_royal_flush(nhole_card, ncom_card):
        result = False
        is_straight_flush, value, colour = ComputeHandStrength.check_if_straight_flush(nhole_card, ncom_card)

        if value == 10:
            result = True

        return result, colour


