

class BaseEvaluator:
    def __init__(self, backbone=None, gt_data=None, **kwargs):


        self.model = backbone
        self.gt_data = gt_data # (B, H, W, C), uint8
        self.preload(**kwargs)

    def preload(self):
        pass

    def run(self):
        raise NotImplementedError("The run method should be implemented in the subclass.")