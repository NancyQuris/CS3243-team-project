class TrainResult:
    result = []

    def add_result(self, r):
        self.result.append(r)

    def get_result(self, part_size):
        i = 0
        avg = []
        while i <= len(self.results) - part_size:
            sum = 0
            for j in range(part_size):
                sum += self.results[i + j]
            avg.append(float(sum) / part_size)
            i += part_size
        return avg

