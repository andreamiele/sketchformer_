import torchmetrics

class MetricManager(object):

    def __init__(self):
        self.metric_names = []
        self.metric_fns = {}

    def add_mean_metric(self, name):
        self.metric_names.append(name)
        self.metric_fns[name] = torchmetrics.MeanMetric()

    def add_sparse_categorical_accuracy(self, name):
        self.metric_names.append(name)
        self.metric_fns[name] = torchmetrics.Accuracy()

    def compute(self, name, *args):
        assert name in self.metric_names, 'Error! {} metric not found.'.format(name)
        # The torchmetrics API requires us to update the state with the new values
        # and then compute the metric after all updates are done.
        # We assume here that args[0] contains the predictions and args[1] contains the labels.
        self.metric_fns[name].update(*args)

    def reset(self):
        for metric_fn in self.metric_fns.values():
            metric_fn.reset()

    def get_results(self):
        # Compute the metric after all updates are done.
        return [metric_fn.compute() for metric_fn in self.metric_fns.values()]

    def get_results_as_dict(self):
        # Compute the metric after all updates are done.
        return {name: self.metric_fns[name].compute().item() for name in self.metric_names}
