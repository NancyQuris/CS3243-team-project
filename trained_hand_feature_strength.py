import numpy as np

class Trained_hand_feature():
    # array length:  18
    street_1 = np.array([0.20846469, 0.19166184, 0.20643419, 0.        , 0.        , 0.        ,
                         0.        , 0.20643419, 0.18700509, 0.        , 0.        , 0.        ,
                         0.        , 0.        , 0.        , 0.        , 0.        , 0.        ], dtype=np.float64)
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

    def get_strength_avg(self):
        result = []
        for d in self.streets:
            result = np.add(d, result)
        result = np.true_divide(result, len(self.streets))
        return result

    def get_street_theta_essential(self, street_idx):
        str = self.streets[street_idx]
        w_val_comb = str[3] + str[4] + str[5] + str[6] + str[7]
        w_suit_comb = str[9] + str[10] + str[11] + str[12] + str[13]
        w_seq_comb = str[15] + str[16] + str[17]
        return [str[0], str[1], str[2], w_val_comb, str[8], w_suit_comb, str[14], w_seq_comb]

    def get_strength_essential_initial(self):
        result = []
        for i in range(4):
            step_theta = dict()
            step_theta['raise'] = np.array([0] + self.get_street_theta_essential(i))
            step_theta['call'] = np.array([0] + self.get_street_theta_essential(i))
            step_theta['fold'] = np.zeros(9)
            result.append(step_theta)
        return result
