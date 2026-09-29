class Vec:

    def __init__(self, elements):
        self.elements = tuple(elements)

    def mean(self):
        return sum(self.elements) / len(self.elements)

    def demean(self):
        mean_value = self.mean()
        return Vec([x - mean_value for x in self.elements])

    def std(self):
        d = self.demean()
        return (sum(x ** 2 for x in d.elements) / len(d.elements)) ** 0.5