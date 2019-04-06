from card_feature import CardFeatureCompute
from card_feature_vector import CardFeatureVectorCompute
import numpy as np


class FeatureStrengthOffline:
    def __init__(self):
        self.cfc = CardFeatureCompute()
        self.cfvc = CardFeatureVectorCompute()
        self.feature_num = self.cfc.feature_num
        self.step_num = 4
        self.feature_prediction_map = np.zeros([self.step_num, self.feature_num])
        self.feature_prediction_map_size = np.zeros(self.step_num)

    def get_step_feature_dict(self, hole, community, step):
        assert((step > 0) and (step < 5))
        feature_dict = self.cfc.fetch_feature(hole, community)
        return feature_dict

    def get_step_feature_vector(self, hole, community, step):
        assert ((step > 0) and (step < 5))
        feature_vector = np.array(self.cfvc.fetch_feature(hole, community))
        return feature_vector

    def feed_bunch_feature_prob_map(self, round_info_list):
        for info in round_info_list:
            self.feed_round_feature_prob_map(info[0], info[1])

    def feed_round_feature_prob_map(self, feature_vector_list, result):
        for i in range(len(feature_vector_list)):
            self.feed_step_feature_prob_map(feature_vector_list[i], result, i)

    def feed_step_feature_prob_map(self, feature_vector_list, result, step):
        vfunc = np.vectorize(self.compare_predict)
        np.add(self.feature_prediction_map[step], vfunc(feature_vector_list, result))

    def output_feature_map(self):
        final_feature_map = []
        for i in range(self.step_num):
            normalised = np.true_divide(self.feature_prediction_map[i], self.feature_prediction_map_size[i])
            final_feature_map.append(normalised)
        return final_feature_map

    def output_weight_suggest(self):
        final_weight_suggest = []
        normalised_feature_p = self.output_feature_map()
        for v in normalised_feature_p:
            sum = np.sum(v)
            normalised = np.true_divide(v, sum)
            final_weight_suggest.append(normalised)
        return final_weight_suggest

    @staticmethod
    def compare_predict(predict, result):
        return result if predict > 0.5 else (1 - result)


