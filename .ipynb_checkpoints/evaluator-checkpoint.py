import pickle
from math import sqrt, ceil
import statistics as st

import visualkeras
from PIL import ImageFont
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import holoviews as hv
from holoviews import opts, dim
from bokeh.io import export_svgs
from selenium import webdriver

from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
import tensorflow as tf

hv.extension('bokeh')
sns.set_theme()


class Evaluator:
    def __init__(self, cm_path: str, labels_path: str, tokenizer_path: str, model_path: str = None, history_path: str = None):
        self.cm: pd.DataFrame = pd.read_csv(filepath_or_buffer=cm_path, index_col=0)

        with open(labels_path, "rb") as f:
            self.labels: list = pickle.load(f)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer: Tokenizer = pickle.load(f)

        if model_path:
            with tf.device("/cpu:0"):
                self.model: Model = load_model(filepath=model_path)

        self.history: pd.DataFrame = pd.read_excel(io=history_path) if history_path else None

        self.binary_cms: dict = self.get_binary_cms()

    def get_model_views(self, save: bool = True):
        try:
            font = ImageFont.truetype("/usr/share/fonts/TTF/Arial.TTF", 32)  # Linux
        except OSError:
            font = ImageFont.truetype("arial.ttf", 32)  # Windows

        color_map = defaultdict(dict)
        color_map[Embedding]['fill'] = '#ff7f7f'
        color_map[Conv1D]['fill'] = '#7fff7f'
        color_map[MaxPooling1D]['fill'] = '#7f7fff'
        color_map[GlobalMaxPooling1D]['fill'] = '#bf7fff'
        color_map[Dense]['fill'] = '#ffff7f'

        return {
            "layered-3d": visualkeras.layered_view(self.model, legend=True, font=font, spacing=100, color_map=color_map, scale_xy=1, scale_z=.8, max_z=1000, to_file="results/model-visualizations/model-layered-3d.png") if save else None,
            "layered-2d": visualkeras.layered_view(self.model, legend=True, font=font, spacing=100, draw_volume=False, color_map=color_map, scale_xy=1, scale_z=.8, max_z=1000, to_file="results/model-visualizations/model-layered-2d.png"  if save else None),
            "graph": visualkeras.graph_view(self.model, color_map=color_map, to_file="results/model-visualizations/model-graph.png"  if save else None)
        }

    def plot_history(self, figsize: tuple):
        """Plots model accuracy evolution, if a model is trained and its history stored."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        sns.lineplot(data=self.history[["acc", "val_acc"]], ax=ax1)
        ax1.set_title('Model accuracy')
        ax1.set(xlabel='epoch', ylabel='accuracy')
        ax1.legend(['train', 'val'], loc='upper left')

        sns.lineplot(data=self.history[["loss", "val_loss"]], ax=ax2)
        ax2.set_title('Model loss')
        ax2.set(xlabel='epoch', ylabel='loss')
        ax2.legend(['train', 'val'], loc='upper left')

        return fig, (ax1, ax2)

    def accuracy(self):
        """Calculates overall accuracy."""

        return sum([self.cm.iloc[i, i] for i in range(len(self.labels))]) / self.cm.sum().sum()

    def get_binary_cms(self) -> dict:
        """Given n labels, generates n 2x2 matrices describing TP, TN, FP, FN of a multiclass confusion matrix."""

        binary_cms = {}

        for label in self.labels:
            # getting the index of label in evaluation list of labels
            i = self.labels.index(label)

            # calculating metrics
            TP = self.cm.iloc[i, i]
            FP = self.cm.iloc[:, i].sum().sum() - TP
            FN = self.cm.iloc[i, :].sum().sum() - TP
            TN = self.cm.sum().sum() - TP - FP - FN

            # construct DataFrame for current label
            binary_cms[label] = pd.DataFrame(
                data={
                    "PREDICTED POSITIVE": [TP, FP],
                    "PREDICTED NEGATIVE": [FN, TN]
                },
                index=["ACTUAL POSITIVE", "ACTUAL NEGATIVE"]
            )

        return binary_cms

    def plot_binary_cm(self, label: str, figsize: tuple, ax: plt.axis = None) -> plt.axis:
        """Given a 2x2 confusion matrix, plot its visualization."""

        individual_cm = self.binary_cms[label]

        names = ["TP", "FN", "FP", "TN"]
        counts = [f"{i}" for i in individual_cm.to_numpy().flatten()]
        actual_pos_percentages = [f"{i / individual_cm.iloc[0, :].sum()*100:.3f}%" for i in individual_cm.iloc[0, :].T.to_numpy().flatten()]
        actual_neg_percentages = [f"{i / individual_cm.iloc[1, :].sum()*100:.3f}%" for i in individual_cm.iloc[1, :].T.to_numpy().flatten()]
        percentages = actual_pos_percentages + actual_neg_percentages

        cm_labels = [f"{n}\n{c}\n{p}" for n, c, p in zip(names, counts, percentages)]
        cm_labels = np.asarray(cm_labels).reshape(2, 2)

        if not ax:
            fig, ax = plt.subplots(1, figsize=figsize)

        sns.heatmap(
            data=individual_cm,
            annot=cm_labels,
            fmt="",
            cmap="Blues",
            linewidths=.5,
            square=True,
            ax=ax,
            cbar=False
        )

        title = label if len(label) < 5*figsize[0] else self.halfsplit(label)
        ax.set_title(f"{title} [{self.get_true_amount(label)}]")

        return ax

    @staticmethod
    def halfsplit(s: str):
        l1 = []
        l2 = s.split(' ')

        while sum([len(x) for x in l1]) < sum([len(x) for x in l2]):
            l1.append(l2.pop(0))

        return ' '.join(l1) + '\n' + ' '.join(l2)

    def save_binary_cms(self, figsize: tuple, path: str):
        """Save png and svg files of binary confusion matrices for each label."""

        for label in self.labels:
            fig, ax = plt.subplots(1, figsize=figsize);
            ax = self.plot_binary_cm(label=label, figsize=figsize, ax=ax);
            filename = self.get_filename(label)
            fig.savefig(f"{path}/{filename}.svg", format="svg")
            # fig.savefig(f"{path}/{filename}.png")

    @staticmethod
    def get_filename(label: str):
        return '-'.join([word.replace(',', '') for word in label.lower().split() if word != '-'])

    def plot_interest_cms(self, labels_of_interest: list, figsize: tuple, hfigs: int = None, hspace: float = None) -> tuple:
        """Given 2x2 confusion matrices of n labels of interest, plot their combined visualization."""

        # calculate number of horizontal (hfigs) and vertical (vfigs) figures
        if hfigs:
            vfigs = ceil(len(labels_of_interest) / hfigs)
        else:
            hfigs = ceil(sqrt(len(labels_of_interest)))
            if len(labels_of_interest) == hfigs*hfigs:
                vfigs = hfigs
            else:
                vfigs = hfigs-1

        fig, axs = plt.subplots(vfigs, hfigs, figsize=figsize)

        subplot_size = figsize[0] / hfigs

        row = 0
        col = 0

        # add subplots of each label of interest to the figure
        for i in range(len(labels_of_interest)):
            self.plot_binary_cm(labels_of_interest[i], figsize=(subplot_size, subplot_size), ax=axs[row, col])

            if col + 1 < hfigs:
                col += 1
            else:
                col = 0
                row += 1

        # removing unused axis
        nfigs = hfigs * vfigs
        col = hfigs - 1

        while nfigs > len(labels_of_interest):
            axs[row, col].set_axis_off()
            nfigs -= 1
            col -= 1

        if hspace:
            fig.subplots_adjust(hspace=hspace)

        return fig, axs

    def get_true_amount(self, label: str) -> int:
        """Get amount of true samples for a given label."""

        i = self.labels.index(label)
        return self.cm.iloc[i, :].sum()

    def get_hits(self, label: str) -> int:
        """Get amount of hits for a given label."""

        i = self.labels.index(label)
        return self.cm.iloc[i, i]

    def get_sensitivity(self, label: str) -> float:
        """Get sensitivity for a given label."""

        return self.get_hits(label) / self.get_true_amount(label)

    def extract_metrics(self, label: str) -> dict:
        """
        Returns evaluation metrics for a given label.\n\n
        P: Positives;\n
        N: Negatives;\n
        TP: True Positives;\n
        TN: True Negatives;\n
        FP: False Positives;\n
        TPR: True Positive Rate, Sensitivity, Recall, or Hit Rate;\n
        TNR: True Negative Rate, Specificity, Selectivity;\n
        PPV: Positive Predictive Value, or Precision;\n
        NPV: Negative Predictive Value;\n
        FNR: False Negative Rate, or Miss Rate;\n
        FPR: False Positive Rate, or Fall-Out;\n
        FDR: False Discovery Rate;\n
        FOR: False Omission Rate;\n
        PLR: Positive Likelihood Ratio;\n
        NLR: Negative Likelihood Ratio;\n
        PT: Prevalence Threshold;\n
        PRE: Prevalence;\n
        ACC: Accuracy;\n
        BA: Balanced Accuracy;\n
        F1: F1 Score;\n
        MCC: Matthews Correlation Coefficient, or Phi Coefficient;\n
        FM: Fowlkes-Mallows index;\n
        BM: Bookmaker Informedness;\n
        MK: Markedness;\n
        DOR: Diagnostic Odds Ratio.
        """

        binary_cm: pd.DataFrame = self.binary_cms[label]

        P: int = binary_cm.loc["ACTUAL POSITIVE"].sum()
        N: int = binary_cm.loc["ACTUAL NEGATIVE"].sum()
        TP: int = binary_cm.loc["ACTUAL POSITIVE", "PREDICTED POSITIVE"]
        FN: int = binary_cm.loc["ACTUAL POSITIVE", "PREDICTED NEGATIVE"]
        FP: int = binary_cm.loc["ACTUAL NEGATIVE", "PREDICTED POSITIVE"]
        TN: int = binary_cm.loc["ACTUAL NEGATIVE", "PREDICTED NEGATIVE"]
        TPR: float = TP / P
        TNR: float = TN / N
        PPV: float = TP / (TP + FP)
        NPV: float = TN / (TN + FN)
        FNR: float = FN / P
        FPR: float = FP / N
        FDR: float = FP / (FP + TP)
        FOR: float = FN / (FN + TN)
        PLR: float = TPR / FPR
        NLR: float = FNR / TNR
        try:
            PT: float = sqrt(FPR) / (sqrt(TPR) + sqrt(FPR))
        except ZeroDivisionError:
            PT = 0
        TS: float = TP / (TP + FN + FP)
        PRE: float = P / (P + N)
        ACC: float = (TP + TN) / (P + N)
        BA: float = (TPR + TNR) / 2
        try:
            F1: float = 2 * (PPV * TPR) / (PPV + TPR)
        except ZeroDivisionError:
            F1 = 0
        MCC: float = sqrt(PPV * TPR * TNR * NPV) - sqrt(FDR * FNR * FPR * FOR)
        FM: float = sqrt(PPV * TPR)
        BM: float = TPR + TNR - 1
        MK: float = PPV + NPV - 1
        DOR: float = PLR / NLR

        return {
            "P": P, "N": N,
            "TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "TPR": TPR, "TNR": TNR,
            "PPV": PPV, "NPV": NPV,
            "FNR": FNR, "FPR": FPR,
            "FDR": FDR, "FOR": FOR,
            "PLR": PLR, "NLR": NLR,
            "PT": PT, "TS": TS,
            "PRE": PRE, "ACC": ACC,
            "BA": BA, "F1": F1,
            "MCC": MCC, "FM": FM,
            "BM": BM, "MK": MK,
            "DOR": DOR
        }

    def generate_report(self) -> pd.DataFrame:
        """Generate complete evaluation report."""

        metrics: list = [
            "P",
            "N",
            "TP",
            "FN",
            "FP",
            "TN",
            "TPR",
            "TNR",
            "PPV",
            "NPV",
            "FNR",
            "FPR",
            "FDR",
            "FOR",
            "PLR",
            "NLR",
            "PT",
            "TS",
            "PRE",
            "ACC",
            "BA",
            "F1",
            "MCC",
            "FM",
            "BM",
            "MK",
            "DOR"
        ]

        reports: dict = {}

        for label in self.labels:
            reports[label]: dict = self.extract_metrics(label)

        general_report: dict = {"label": self.labels}

        for metric in metrics:
            general_report[metric]: list = [reports[label][metric] for label in reports.keys()]

        return pd.DataFrame(general_report).fillna(0)

    def plot_best(self, report: pd.DataFrame, metric: str, top: int, figsize: tuple, title: str = None, keep_perfect: bool = True) -> tuple:
        """Plot the top best labels for a given metric."""

        data: pd.DataFrame = report.sort_values(by=[metric], ascending=False)

        if not keep_perfect:
            data = data.drop(data.loc[data[metric] == 1].index)

        data = data.iloc[:top][["label", metric]]

        fig, ax = plt.subplots(1, figsize=figsize)
        sns.barplot(x=data[metric], y=data["label"])
        plt.title(title if title else f"Top {top} {metric}")

        return fig, ax

    def plot_chord(self, same: bool = False, max_true: int = 0, figdim: int = 1000, fontfigratio: float = 0.01, interest: list = None, savepath: str = None) -> hv.Chord:
        """Plots a chord visualization of the relationship between predictions."""

        if interest:
            # selects labels from a list
            indexes = [self.labels.index(label) for label in interest]
            cm = self.cm.loc[indexes]
            cm = cm[[str(i) for i in indexes]]
        else:
            # selects labels with sample amount greater than max_true
            cm = self.cm.loc[self.cm.sum(axis=1) > max_true]
            cm = cm[[str(i) for i in cm.index]]

        # construct link dataframe for the chord diagram
        data = {
            "source": [],
            "target": [],
            "value": []
        }

        for index in cm.columns:
            predicted = cm[index]
            predicted = predicted.loc[predicted > 0]

            for item in predicted.iteritems():
                add = True

                if not same:
                    if item[0] == int(index):
                        add = False

                if add:
                    data["source"].append(int(index))
                    data["target"].append(item[0])
                    data["value"].append(item[1])

        links = pd.DataFrame(data)

        # construct nodes dataframe for the chord diagram
        nodes = pd.DataFrame({"name": self.labels})
        nodes = nodes.iloc[cm.index].reset_index()
        nodes = hv.Dataset(nodes, 'index')

        # construct chord diagram using links and nodes
        chord = hv.Chord((links, nodes)).select(value=(5, None))
        chord.opts(
            opts.Chord(
                fontsize={'labels': figdim*fontfigratio},
                height=figdim, width=figdim,
                cmap='Category20',
                edge_cmap='Category20',
                edge_color=dim('source').str(),
                labels='name',
                node_color=dim('index').str()
            )
        )

        if savepath:
            p = hv.render(chord, backend="bokeh")
            p.output_backend = "svg"
            browser = webdriver.Firefox(executable_path="/usr/local/bin/geckodriver")
            export_svgs(p, filename=f"{savepath}/chord.svg", webdriver=browser)

        return chord

    def plot_sankey(self, label: str, quantile: int = None, figdim: int = 1000, fontfigratio: float = 0.01, counts: bool = False) -> go.Figure:
        """Plots a sankey flow visualization of predicted labels for a given label."""

        # selects series of samples for a label, distributed by its predictions
        index = self.labels.index(label)
        predicted: pd.DataFrame = self.cm.iloc[index, :]
        predicted = predicted.set_axis(self.labels)
        predicted = predicted[predicted > 0]

        predicted = predicted.sort_values(ascending=False)

        # groups labels under the calculated quantile into a label named OTHERS
        if quantile:
            others: pd.DataFrame = predicted.loc[predicted < st.quantiles(list(predicted.values))[quantile]]
            if label in list(predicted.index):
                if predicted[label] < st.quantiles(list(predicted.values))[quantile]:
                    others = others.drop(label)
            predicted = predicted.drop(others.index)
            predicted = predicted.append(pd.Series([others.sum()], index=["OTHERS"]))

        predicted = predicted.sort_values(ascending=False)

        # create colors and inserting color for current label at front
        colors = [f"rgba({', '.join([str(c) for c in color])}, 0.8)" for color in sns.color_palette("hls", len(predicted))]

        # construct sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=[f"{label} [{predicted.sum()}]"] + [f"{item[0]} [{item[1]}]" for item in predicted.iteritems()] if counts else [label] + list(predicted.index),
                        color=[f"rgba({', '.join([str(c) for c in sns.color_palette('hls', len(predicted))[0]])}, 0.8)"] + colors
                    ),
                    link=dict(
                        source=[0 for i in range(len(predicted))],
                        target=[i + 1 for i in range(len(list(predicted.index)))],
                        value=[p for p in predicted],
                        color=colors
                    )
                )
            ]
        )
        fig.update_layout(
            title_text=f"Predictions for {label if len(label) < 50 else self.halfsplit(label)}",
            title_x=0.5,
            font_size=figdim*fontfigratio,
            autosize=False,
            width=figdim,
            height=figdim
        )

        return fig

    def save_sankeys(self, path: str, quantile: int = None, fontfigratio: float = 0.01, counts: bool = False):
        """Save png and svg files of sankey diagrams for each label."""

        for label in self.labels:
            try:
                fig = self.plot_sankey(label=label, quantile=quantile, fontfigratio=fontfigratio, counts=counts)
                filename = self.get_filename(label)
                fig.write_image(f"{path}/{filename}.svg")
                # fig.write_image(f"{path}/{filename}.png")
            except st.StatisticsError:
                print(f"Could not create sankey for {label} [{self.cm.iloc[self.labels.index(label)].sum()}]. Must have at least two data points.")
