from processing.model.gaze.tester import GazeTester
from processing.utils.early_stopping import EarlyStopping


class GazeEarlyStopping(EarlyStopping):
    def __init__(self, cf, model, val_dataloader, dir, device, task, monitor, monitor_mode):
        tester = GazeTester(model, val_dataloader, device, task)
        super().__init__(cf, model, dir, monitor, monitor_mode, tester)
