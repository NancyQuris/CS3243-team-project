from card_feature import CardFeatureCompute
from card_feature_vector import CardFeatureVectorCompute
import numpy as np

class TrainStrength:
    cfc = CardFeatureCompute()
    cfvc = CardFeatureVectorCompute()

    feature_num = 0
    step_num = 4
    mu = 0.01  # the small step to adjust
    threshold = 1 #above threshold: predict success

    weights = np.empty([1, 1])
    step_scale = np.empty(1)
    round_scale_factor = [1, 0.8, 0.6, 0.5]

    def __init__(self):
        f_num = self.cfc.feature_num
        s_num = self.step_num
        self.feature_num = f_num
        self.weights = np.full((s_num, f_num), 1 / f_num)
        self.step_scale = np.full(s_num, 1 / s_num)

    def get_strength(self, hole, community, step):
        assert((step > 0) and (step < 5))
        feature_dict = self.cfc.fetch_feature(hole, community)
        feature_vector = np.array(feature_dict.values())
        weight = self.weights[step - 1]
        strength = np.dot(feature_vector, weight) * self.step_scale[step - 1]
        return strength, feature_vector

    def get_strength_by_vector(self, hole, community, step):
        assert ((step > 0) and (step < 5))
        feature_vector = np.array(self.fvc.fetch_feature(hole, community))
        weight = self.weights[step - 1]
        strength = np.dot(feature_vector, weight) * self.step_scale[step - 1]
        return strength, feature_vector


    def get_feature_prob_map(self, trial_num, feature_vector_list, result_list):
        total_hit_count = np.full(self.feature_num, 0)
        for i in range(trial_num):
            vfunc = np.vectorize(self.compare_predict())
            np.add(total_hit_count, vfunc(feature_vector_list[i], result_list[i]))
        return np.true_divide(total_hit_count, trial_num)

    def adjust_weights(self, trial_num, feature_vector_list, result_list):
        prob_map = self.get_feature_prob_map(trial_num, feature_vector_list, result_list)



    @staticmethod
    def compare_predict(predict, result):
        return result if predict > 0.5 else (1 - result)


