"""
# ------------------------------------------------------------------------------------#
#  Distributed Multidisciplinary Design Optimization - DMDO                           #
#                                                                                     #
#  Author: Ahmed H. Bayoumy                                                           #
#  email: ahmed.bayoumy@mail.mcgill.ca                                                #
#                                                                                     #
#  This program is free software: you can redistribute it and/or modify it under the  #
#  terms of the GNU General Public License as published by the Free Software          #
#  Foundation, either version 3 of the License, or (at your option) any later         #
#  version.                                                                           #
#                                                                                     #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY    #
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A    #
#  PARTICULAR PURPOSE.  See the GNU General Public License for more details.          #
#                                                                                     #
#  You should have received a copy of the GNU General Public License along            #
#  with this program. If not, see <http://www.gnu.org/licenses/>.                     #
#                                                                                     #
#  You can find information on simple_mads at                                         #
#  https://github.com/Ahmed-Bayoumy/DMDO                                              #
# ------------------------------------------------------------------------------------#
# """

import numpy as np
import matplotlib.pyplot as plt
import os

from DMDO.ai.detector import HybridAnomalyDetector

import pandas as pd
# =========================================================
# SIGNAL GENERATORS
# =========================================================

def generate_flat(n):
    return np.ones(n) * 10, np.ones(n) * 1e-2


def generate_spiky(n):
    x = np.ones(n) * 10
    y = np.ones(n) * 1e-2

    spikes = np.random.choice(np.arange(10, n), size=10, replace=False)
    for s in spikes:
        y[s] *= np.random.uniform(10, 100)

    return x, y


def generate_mixed(n):
    x = np.ones(n) * 10
    y = np.ones(n) * 1e-2

    for i in range(n):
        if i % 25 == 0:
            y[i] *= np.random.uniform(15, 120)
        else:
            y[i] *= (1 + 0.05 * np.random.randn())

    return x, y


def generate_drift(n):
    return np.linspace(10, 20, n), np.exp(np.linspace(-4, -1, n))


def generate_degrading(n):
    obj = np.linspace(10, 20, n)   # worsening objective
    inc = np.ones(n) * 1e-2

    spikes = np.random.choice(np.arange(10, n), size=10, replace=False)
    for s in spikes:
        inc[s] *= np.random.uniform(20, 150)

    return obj, inc


def generate_improving(n):
    obj = np.linspace(20, 10, n)   # improving objective
    inc = np.ones(n) * 1e-2

    spikes = np.random.choice(np.arange(10, n), size=10, replace=False)
    for s in spikes:
        inc[s] *= np.random.uniform(20, 150)

    return obj, inc

def data_hist(n=200, datafile: str = None):
    
    COLUMN_NAME_OBJ = "                     Objective"
    COLUMN_NAME_INC = "            Max. inconsistency"
    if datafile is not None:
        df = pd.read_csv(datafile, sep=None, engine='python')
        data_obj = df[COLUMN_NAME_OBJ].astype(float).values
        data_inc = df[COLUMN_NAME_INC].astype(float).values
        hmin_columns = []
        for column in df.columns:
            label = str(column).strip()
            if label.startswith("hmin_"):
                try:
                    hmin_columns.append((int(label.split("_")[-1]), column))
                except ValueError:
                    continue
        hmin_columns.sort(key=lambda item: item[0])
        hmin_hist = []
        for _, row in df.iterrows():
            if hmin_columns:
                hmin_hist.append([float(row[column]) for _, column in hmin_columns])
            else:
                hmin_hist.append([])
        return data_obj, data_inc, hmin_hist
    return np.array([]), np.array([]), []

# =========================================================
# TRUE SPIKES (BASE)
# =========================================================

def true_spikes(signal):
    x = np.log10(np.clip(signal, 1e-12, None))

    dx = np.diff(x, prepend=x[0])
    ddx = np.diff(dx, prepend=dx[0])

    score = 0.6 * np.abs(ddx) + 0.4 * np.abs(dx)

    thr = np.mean(score) + 2.0 * np.std(score)

    return (score > thr).astype(int)


# =========================================================
# OBJECTIVE-AWARE TRUE LABELS
# =========================================================

def true_spikes_objective(signal, obj, mode="neutral"):

    base = true_spikes(signal)

    obj = np.asarray(obj)
    obj_trend = np.diff(obj, prepend=obj[0])

    if mode == "degrading":
        # spikes during worsening objective are more important
        weight = 1 + (obj_trend > 0).astype(float)

    elif mode == "improving":
        # spikes during improvement are less important
        weight = 1 - 0.5 * (obj_trend < 0).astype(float)

    else:
        weight = np.ones_like(base, dtype=float)

    return (base * weight > 0.5).astype(int)


# =========================================================
# METRICS
# =========================================================

def compute_metrics(true, pred):

    L = min(len(true), len(pred))
    true = np.asarray(true[:L])
    pred = np.asarray(pred[:L])

    pred_bin = (pred > 0).astype(int)

    TP = np.sum((true == 1) & (pred_bin == 1))
    FP = np.sum((true == 0) & (pred_bin == 1))
    FN = np.sum((true == 1) & (pred_bin == 0))

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1, TP, FP, FN


# =========================================================
# PLOTTING
# =========================================================

def plot_results(name, inc, obj, true, detector):

    pred = np.array(detector.alarm_history)
    prob = np.array(detector.prob_history)
    unc = np.array(detector.unc_history)
    forecast_prob = np.array(detector.forecast_prob_history)
    non_improvement_prob = np.array(detector.non_improvement_prob_history)
    objective_non_improvement_prob = np.array(detector.objective_non_improvement_prob_history)
    constraint_non_improvement_prob = np.array(detector.constraint_non_improvement_prob_history)

    L = min(len(inc), len(true), len(pred))

    inc = inc[:L]
    true = true[:L]
    pred = pred[:L]
    prob = prob[:L]
    unc = unc[:L]
    forecast_prob = forecast_prob[:L]
    non_improvement_prob = non_improvement_prob[:L]
    objective_non_improvement_prob = objective_non_improvement_prob[:L]
    constraint_non_improvement_prob = constraint_non_improvement_prob[:L]

    true_idx = np.where(true == 1)[0]
    pred_idx = np.where(pred > 0)[0]

    fig, axs = plt.subplots(5, 1, figsize=(12, 13))

    # Signal on primary Y-axis (left)
    axs[0].plot(inc, label="Signal", color='tab:blue')
    axs[0].scatter(true_idx, inc[true_idx], c="green", label="True spikes")
    axs[0].scatter(pred_idx, inc[pred_idx], c="red", label="Predicted alarms")
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Max Inc. (log)")

    # Objective on secondary Y-axis (right)
    ax0_twin = axs[0].twinx() 
    ax0_twin.plot(obj, label="Objective", color='tab:orange', linestyle='--')
    ax0_twin.set_yscale("log")
    ax0_twin.set_ylabel("Objective (log)")

    # Combine legends from both axes
    lines, labels = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2, loc='upper left')
    
    axs[0].set_title(name)

    axs[1].plot(pred, label="Alarm level")
    axs[1].legend()

    axs[2].plot(prob, label="Spike probability")
    axs[2].plot(forecast_prob, label="Forecast spike probability", linestyle='--')
    axs[2].fill_between(range(L),
                        np.clip(prob - unc, 0.0, 1.0),
                        np.clip(prob + unc, 0.0, 1.0),
                        alpha=0.2)
    axs[2].legend()

    axs[3].plot(non_improvement_prob, label="Spike to non-improvement")
    axs[3].plot(objective_non_improvement_prob, label="Objective non-improvement")
    axs[3].plot(constraint_non_improvement_prob, label="Constraint non-improvement")
    axs[3].legend()

    axs[4].plot(unc, label="Uncertainty")
    axs[4].legend()

    plt.tight_layout()
    plt.show()


# =========================================================
# TEST RUNNER (FULL FIXED BLOCK YOU ASKED FOR)
# =========================================================

def run_test(name, fn, N=200):

    print(f"\n🚀 Running test: {name}")

    detector = HybridAnomalyDetector(window=6, min_history=5, model_type="LSTM", seed=12345)
    hmin_hist = None

    if name == "data_hist":
        history_file = os.path.join(os.path.dirname(__file__), "SBJ", "Coordination_history.out")
        obj, inc, hmin_hist = fn(N, history_file)
    else:
        obj, inc = fn(N)

    preds = []

    for idx, (o, i) in enumerate(zip(obj, inc)):
        hmins = hmin_hist[idx] if hmin_hist is not None and idx < len(hmin_hist) else None
        preds.append(detector.update(o, i, hmins=hmins))

    # =========================
    # 🔥 CORRECTED CORE LOGIC (RESTORED EXACTLY)
    # =========================
    if name == "degrading":
        true = true_spikes_objective(inc, obj, mode="degrading")

    elif name == "improving":
        true = true_spikes_objective(inc, obj, mode="improving")
    elif name == "data_hist":
        true = true_spikes_objective(inc, obj, mode="neutral")
    else:
        true = true_spikes(inc)

    # =========================
    # METRICS
    # =========================

    precision, recall, f1, TP, FP, FN = compute_metrics(true, preds)

    print("\n📊 VALIDATION RESULTS")
    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    # =========================
    # PASS / FAIL RULES
    # =========================

    if name == "flat":
        passed = f1 < 0.2

    elif name == "spiky":
        passed = (f1 >= 0.45) and (recall >= 0.4)

    elif name == "mixed":
        passed = (0.3 <= f1 <= 0.7) and (recall >= 0.3)

    elif name == "drift":
        passed = f1 < 0.2

    elif name == "degrading":
        passed = (f1 >= 0.45) and (recall >= 0.5)

    elif name == "improving":
        passed = f1 < 0.2 and precision < 0.2

    else:
        passed = True

    print("\n🧪 RESULT:", "PASS ✅" if passed else "FAIL ❌")

    # =========================
    # VISUALIZATION
    # =========================

    plot_results(name, inc, obj, true, detector)


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":

    run_test("flat", generate_flat)
    run_test("spiky", generate_spiky)
    run_test("mixed", generate_mixed)
    run_test("drift", generate_drift)
    run_test("degrading", generate_degrading)
    run_test("improving", generate_improving)
    # run_test("data_hist", data_hist)