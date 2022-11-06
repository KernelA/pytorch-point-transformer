from matplotlib import pyplot as plt
from typing import Dict


def plot_conf_matrix(conf_matrix, class_mapping: Dict[str, int]):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_title("Confusion matrix")
    ax.imshow(conf_matrix, interpolation="none", cmap="summer_r")

    labels = tuple(name for name, _ in sorted(class_mapping.items(), key=lambda x: x[1]))
    num_labels = tuple(range(len(labels)))
    ax.set_xticks(num_labels)
    ax.set_xticklabels(labels)
    ax.set_yticks(num_labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in num_labels:
        for j in num_labels:
            _ = ax.text(j, i, "{0:.4f}".format(conf_matrix[i, j]),
                           ha="center", va="center", color="black")

    return fig
