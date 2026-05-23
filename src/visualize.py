# =====================================================
# visualize.py
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# =====================================================
# LOAD SAVED RESULTS
# =====================================================

y_true=np.load(
    "results_true.npy"
)

y_pred=np.load(
    "results_pred.npy"
)

y_prob=np.load(
    "results_prob.npy"
)

vgae_losses=np.load(
    "vgae_losses.npy"
)

rgcn_losses=np.load(
    "rgcn_losses.npy"
)

fusion_losses=np.load(
    "fusion_losses.npy"
)


print("Results loaded")


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm=confusion_matrix(

    y_true,

    y_pred

)

disp=ConfusionMatrixDisplay(

    confusion_matrix=cm

)

disp.plot()

plt.title(
    "Confusion Matrix"
)

plt.savefig(
    "confusion_matrix.png"
)

plt.show()


# =====================================================
# ROC CURVE
# =====================================================

fpr,tpr,_=roc_curve(

    y_true,

    y_prob

)

roc_auc=auc(

    fpr,

    tpr

)

plt.figure()

plt.plot(

    fpr,

    tpr,

    label=f"AUC={roc_auc:.3f}"

)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.savefig(
    "roc_curve.png"
)

plt.show()


# =====================================================
# VGAE LOSS
# =====================================================

plt.figure()

plt.plot(
    vgae_losses
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "VGAE Loss Curve"
)

plt.savefig(
    "vgae_loss.png"
)

plt.show()


# =====================================================
# RGCN LOSS
# =====================================================

plt.figure()

plt.plot(
    rgcn_losses
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "RGCN Loss Curve"
)

plt.savefig(
    "rgcn_loss.png"
)

plt.show()


# =====================================================
# FUSION LOSS
# =====================================================

plt.figure()

plt.plot(
    fusion_losses
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Fusion Loss Curve"
)

plt.savefig(
    "fusion_loss.png"
)

plt.show()


print("\nAll images saved successfully")