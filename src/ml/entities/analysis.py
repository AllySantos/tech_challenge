class AnalysisResult:
    def __init__(self, recall: float, precision: float, cost: float, accuracy: float, f1: float):
        self.recall = recall
        self.precision = precision
        self.cost = cost
        self.accuracy = accuracy
        self.f1 = f1
