


class Comparison:
    def __init__(self, log_reg_fpr, log_reg_fpr_dem, log_reg_metrics, perspective_fpr, persepctive_fpr_dem, perspective_metrics):
        self.log_reg_fpr = log_reg_fpr
        self.log_reg_fpr_dem = log_reg_fpr_dem
        self.log_reg_metrics = log_reg_metrics
        self.perspective_fpr = perspective_fpr
        self.perspective_fpr_dem = persepctive_fpr_dem
        self.perspective_metrics = perspective_metrics

    def compare(self):
        """
        Prints out metrics comparatively between the two models.
        """
        log_reg_fpr_white = self.log_reg_fpr_dem['White']
        log_reg_fpr_hispanic = self.log_reg_fpr_dem['Hispanic']
        log_reg_fpr_aa = self.log_reg_fpr_dem['AA']
        log_reg_fpr_other = self.log_reg_fpr_dem['Other']

        log_reg_accuracy = self.log_reg_metrics['accuracy']
        log_reg_precision = self.log_reg_metrics['precision']
        log_reg_recall = self.log_reg_metrics['recall']
        log_reg_fscore = self.log_reg_metrics['fscore']

        perspective_accuracy = self.perspective_metrics['accuracy']
        perspective_precision = self.perspective_metrics['precision']
        perspective_recall = self.perspective_metrics['recall']
        perspective_fscore = self.perspective_metrics['fscore']
        

        perspective_fpr_white = self.perspective_fpr_dem['White']
        perspective_fpr_hispanic = self.perspective_fpr_dem['Hispanic']
        perspective_fpr_aa = self.perspective_fpr_dem['AA']
        perspective_fpr_other = self.perspective_fpr_dem['Other']

        print("Logistic Regression FPR(White): %f | VS | Perspective Classifier FPR(White): %f" % (log_reg_fpr_white, perspective_fpr_white))
        print("-----------------------------------------------------------------------------------")
        print("Logistic Regression FPR (Hispanic): %f | VS | Perspective Classifier FPR(Hispanic): %f" % (log_reg_fpr_hispanic, perspective_fpr_hispanic))
        print("-----------------------------------------------------------------------------------")
        print("Logistic Regression FPR (AA): %f | VS | Perspective Classifier FPR(AA): %f" % (log_reg_fpr_aa, perspective_fpr_aa))
        print("-----------------------------------------------------------------------------------")
        print("Logistic Regression FPR (Other): %f | VS | Perspective Classifier FPR(Other): %f" % (log_reg_fpr_other, perspective_fpr_other))
        print("-----------------------------------------------------------------------------------")
        print("Logistic Regression Accuracy: %f | VS | Perspective Classifier Accuracy: %f" % (log_reg_accuracy, perspective_accuracy))
        print("-----------------------------------------------------------------------------------")
        print("Logistic Regression Precision: %f | VS | Perspective Classifier Precision: %f" % (log_reg_precision, perspective_precision))
        print("-----------------------------------------------------------------------------------")
        print("Logistic Regression Recall: %f | VS | Perspective Classifier Recall: %f" % (log_reg_recall, perspective_recall))
        print("-----------------------------------------------------------------------------------")
        print("Logistic Regression F-Score: %f | VS | Perspective Classifier F-Score: %f" % (log_reg_fscore, perspective_fscore))