import pandas as pd

class MetricasClassificacao:
    def __init__(self):
        pass
    
    def calcular_acuracia(self, tp, fp, fn, tn):
        # Calcular a acurácia
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        return accuracy

    def calcular_precisao(self, tp, fp):
        # Calcular a precisão
        precision = tp / (tp + fp)
        return precision

    def calcular_recall(self, tp, fn):
        # Calcular a sensibilidade (recall)
        recall = tp / (tp + fn)
        return recall

    def calcular_especificidade(self, fp, tn):
        # Calcular a especificidade
        specificity = tn / (fp + tn)
        return specificity

    def calcular_f1_score(self, precision, recall):
        # Calcular o F1-score
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def calcular_metricas(self, df):
        results = []
        for index, row in df.iterrows():
            tp = row[index]
            fp = row.sum() - tp
            fn = df[index].sum() - tp
            tn = df.values.sum() - tp - fp - fn
            
            accuracy = self.calcular_acuracia(tp, fp, fn, tn)
            precision = self.calcular_precisao(tp, fp)
            recall = self.calcular_recall(tp, fn)
            specificity = self.calcular_especificidade(fp, tn)
            f1_score = self.calcular_f1_score(precision, recall)
            
            results.append([accuracy, precision, recall, specificity, f1_score])
        
        metrics_df = pd.DataFrame(results, columns=['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score'])
        return metrics_df