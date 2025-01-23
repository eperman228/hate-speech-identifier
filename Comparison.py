


class Comparison:
    def __init__(self, log_reg_fpr, log_reg_fpr_dem, log_reg_metrics, perspective_fpr, persepctive_fpr_dem, perspective_metrics):
        self.log_reg_fpr = log_reg_fpr
        self.log_reg_fpr_dev = log_reg_fpr_dem
        self.log_reg_metrics = log_reg_metrics
        self.perspective_fpr = perspective_fpr
        self.perspective_fpr_dev = persepctive_fpr_dem
        self.perspective_metrics = perspective_metrics

    def compare(self):
        """
        Prints out metrics comparatively between the two models.
        """
        print("Logistic Regression FPR (False Positive Rate): ", self.log_reg_fpr)
        print("Perspective API FPR: ", self.perspective_fpr)

        print("Logistic Regression FPR Dev: ", self.log_reg_fpr_dev)
        print("Perspective API FPR Dev: ", self.perspective_fpr_dev)


        print("Logistic Regression Metrics: " + self.log_reg_metrics + "Perspective API Metrics:" + self.perspective_metrics)
