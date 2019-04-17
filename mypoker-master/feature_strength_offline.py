from card_feature import CardFeatureCompute
from card_feature_vector import CardFeatureVectorCompute
import numpy as np
from pypokerengine.engine.card import Card


class FeatureStrengthOffline:
    step_map = {
        0: 0,
        3: 1,
        4: 2,
        5: 3
    }

    def __init__(self):
        self.cfc = CardFeatureCompute()
        self.cfvc = CardFeatureVectorCompute()
        self.feature_num = self.cfc.feature_num
        self.step_num = 4
        self.feature_prediction_map = np.zeros([self.step_num, self.feature_num])
        self.feature_frequency_map = np.zeros([self.step_num, self.feature_num])
        # consists of 2 elements: 1. a list of 4 step vectors 2. round result [0/1]
        self.info_list = []
        # consists of 4 step vectors
        self.step_list = []

    def raw_feed(self, hole, community, step):
        # step represent the number of community card: 0,3,4,5
        assert (step in [0, 3, 4, 5])
        hole_list = []
        community_list = []
        step_true = self.step_map[step]
        for card in hole:
            hole_list.append(Card.from_str(card))
        for card in community:
            community_list.append(Card.from_str(card))
        feature_vector = self.get_step_feature_vector(hole_list, community_list, step_true)
        self.step_list.append(feature_vector)

    def feed_result(self, result):
        assert(len(self.step_list) < 5)
        self.info_list.append([self.step_list, result])
        self.step_list = []

    def get_step_feature_dict(self, hole, community, step):
        assert((step >= 0) and (step < 4))
        feature_dict = self.cfc.fetch_feature(hole, community)
        return feature_dict

    def get_step_feature_vector(self, hole, community, step):
        assert((step >= 0) and (step < 4))
        feature_vector = np.array(self.cfvc.fetch_feature(hole, community))
        return feature_vector

    def feed_bunch_feature_prob_map(self, round_info_list):
        for info in round_info_list:
            self.feed_round_feature_prob_map(info[0], info[1])

    def feed_round_feature_prob_map(self, feature_vector_list, result):
        assert(len(feature_vector_list) < 5)
        for i in range(len(feature_vector_list)):
            self.feed_step_feature_prob_map(feature_vector_list[i], result, i)

    def feed_step_feature_prob_map(self, feature_vector, result, step):
        vfunc_f = np.vectorize(self.check_if_present)
        vfunc_p = np.vectorize(self.compare_predict)
        self.feature_frequency_map[step] = np.add(self.feature_frequency_map[step], vfunc_f(feature_vector))
        self.feature_prediction_map[step] = np.add(self.feature_prediction_map[step], vfunc_p(feature_vector, result))

    def feed_self_feature_prob_map(self):
        self.feed_bunch_feature_prob_map(self.info_list)

    def output_feature_map(self):
        final_feature_map = []
        for i in range(self.step_num):
            f = self.feature_frequency_map[i]
            p = self.feature_prediction_map[i]
            feature_step_map = np.true_divide(p, f, out=np.zeros_like(p), where=f != 0)
            final_feature_map.append(feature_step_map)
        return final_feature_map

    def output_weight_suggest(self):
        final_weight_suggest = []
        normalised_feature_p = self.output_feature_map()
        for v in normalised_feature_p:
            sum = np.sum(v)
            if sum == 0:
                final_weight_suggest.append(np.zeros(self.feature_num))
            else:
                normalised = np.true_divide(v, sum)
                final_weight_suggest.append(normalised)
        return final_weight_suggest

    @staticmethod
    def check_if_present(predict):
        return predict > 0.5

    @staticmethod
    def compare_predict(predict, result):
        return result if predict > 0.5 else 0


