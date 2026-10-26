from pathlib import Path
import argparse
import json
import platform
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import sklearn
import xgboost
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, DMatrix

SEED = 42
CHEMISTRY = ["C","Si","Mn","P","S","Ni","Cr","Cu","Mo"]
HEAT = ["NT","THT","THt","THQCr","CT","Ct","DT","Dt","QmT","TT","Tt","TCr"]
EXTRA = ["RedRatio","dA","dB","dC"]
FEATURES = HEAT + CHEMISTRY + EXTRA
TARGET = "Fatigue Strength"
CANONICAL = ["Sl. No."] + FEATURES + [TARGET]
XGB_PARAMS = dict(objective="reg:squarederror",n_estimators=450,max_depth=3,learning_rate=0.03,subsample=0.85,colsample_bytree=0.85,min_child_weight=1,reg_alpha=0.0,reg_lambda=3.0,tree_method="hist",random_state=SEED,n_jobs=1,verbosity=0)
RF_ESTIMATORS = 700
BASELINE_REPEATS = 10
BOOTSTRAP_REPEATS = 10000


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--nrep-size-matched", type=int, default=20)
    parser.add_argument("--quick-test", action="store_true")
    args, _ = parser.parse_known_args()
    return args


def resolve_data_path(value):
    if value:
        p = Path(value).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(str(p))
    candidates = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / "Fatigue strength dataset.xlsx")
    candidates.append(Path.cwd() / "Fatigue strength dataset.xlsx")
    candidates.append(Path.home() / "Downloads" / "Fatigue strength dataset.xlsx")
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Fatigue strength dataset.xlsx was not found. Place it beside this script or pass --data PATH.")


def ensure_dirs(root):
    dirs = {}
    for name in ["Tables","Figures","Supplementary","Logs"]:
        p = root / name
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    return dirs


def load_dataset(path, root):
    warnings.filterwarnings("ignore", message="Unknown extension is not supported and will be removed")
    xls = pd.ExcelFile(path)
    sheet = "nims.csv" if "nims.csv" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    if df.shape != (437, 27):
        raise ValueError(f"Expected the frozen NIMS benchmark shape (437, 27), found {df.shape}.")
    df.columns = CANONICAL
    for c in FEATURES + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    clean = root / "NIMS_Fatigue_Benchmark_Canonical.xlsx"
    df.to_excel(clean, index=False, sheet_name="nims.csv")
    return df, sheet, clean


def xgb_model():
    return XGBRegressor(**XGB_PARAMS)


def save_final_model(df, output_root):
    import json

    model = xgb_model()

    # Train the final model on the complete dataset.
    X = df[FEATURES]
    y = df[TARGET]

    print("Training final model on all records...")
    model.fit(X, y)

    # Save the trained XGBoost model.
    model_path = output_root / "fatigue_strength_xgb.json"
    model.save_model(str(model_path))

    # Save the information required for future inference.
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "model_type": "XGBRegressor",
        "xgboost_parameters": XGB_PARAMS,
        "training_records": len(df),
        "seed": SEED
    }

    metadata_path = output_root / "fatigue_strength_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")

    return model


def metric_dict(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return dict(R2=float(r2_score(y,p)),MAE_MPa=float(mean_absolute_error(y,p)),RMSE_MPa=float(mean_squared_error(y,p)**0.5),Median_AE_MPa=float(np.median(np.abs(y-p))))


def bootstrap_mean_ci(values, seed=SEED, repeats=None):
    values = np.asarray(values, float)
    repeats = BOOTSTRAP_REPEATS if repeats is None else repeats
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, float)
    for i in range(repeats):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return np.quantile(means, [0.025, 0.975])


def process_routes(df):
    th = (df["THt"].fillna(0) > 0).astype(int)
    c = (df["Ct"].fillna(0) > 0).astype(int)
    d = (df["Dt"].fillna(0) > 0).astype(int)
    t = (df["Tt"].fillna(0) > 0).astype(int)
    return "TH" + th.astype(str) + "_C" + c.astype(str) + "_D" + d.astype(str) + "_T" + t.astype(str)


def composition_geometry(df):
    tuples = [tuple(r) for r in df[CHEMISTRY].to_numpy(float)]
    unique = list(dict.fromkeys(tuples))
    mapping = {u:i for i,u in enumerate(unique)}
    gid = np.array([mapping[x] for x in tuples], int)
    sizes = pd.Series(gid).value_counts().sort_index()
    repeated = set(sizes[sizes > 1].index)
    eligible = np.array([g in repeated for g in gid], bool)
    u = np.asarray(unique, float)
    scaler = StandardScaler().fit(u)
    uz = scaler.transform(u)
    distances = cdist(uz, uz)
    np.fill_diagonal(distances, np.inf)
    nn = distances.min(axis=1)
    q50 = float(np.quantile(nn, 0.50))
    q75 = float(np.quantile(nn, 0.75))
    return tuples, unique, gid, sizes, eligible, scaler, uz, nn, q50, q75


def dataset_audit(df, unique, sizes, eligible):
    return pd.DataFrame({"Metric":["Total records","Predictors","Unique exact compositions","Repeated composition groups","Records in repeated groups","Singleton compositions"],"Value":[len(df),len(FEATURES),len(unique),int((sizes>1).sum()),int(eligible.sum()),int((sizes==1).sum())]})


def run_baseline_robustness(df, tables):
    x = df[FEATURES]
    y = df[TARGET]
    models = {
        "Ridge": Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",Ridge(alpha=1.0))]),
        "Random_Forest": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=RF_ESTIMATORS,max_features="sqrt",min_samples_leaf=1,random_state=SEED,n_jobs=-1))]),
        "XGBoost": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",XGBRegressor(**XGB_PARAMS))])
    }
    splitter = RepeatedKFold(n_splits=5, n_repeats=BASELINE_REPEATS, random_state=SEED)
    rows = []
    for model_name, estimator in models.items():
        for fold, (tr, te) in enumerate(splitter.split(x), 1):
            model = clone(estimator)
            model.fit(x.iloc[tr], y.iloc[tr])
            pred = model.predict(x.iloc[te])
            m = metric_dict(y.iloc[te].to_numpy(), pred)
            rows.append({"Model":model_name,"Fold_Run":fold,**m})
    folds = pd.DataFrame(rows)
    summary = folds.groupby("Model").agg(R2_Mean=("R2","mean"),R2_SD=("R2","std"),MAE_Mean_MPa=("MAE_MPa","mean"),MAE_SD_MPa=("MAE_MPa","std"),RMSE_Mean_MPa=("RMSE_MPa","mean"),RMSE_SD_MPa=("RMSE_MPa","std")).reset_index()
    folds.to_csv(tables / "02_Baseline_Robustness_Fold_Metrics.csv", index=False)
    summary.to_csv(tables / "03_Baseline_Robustness_Summary.csv", index=False)
    xgb_folds = folds[folds.Model == "XGBoost"].drop(columns="Model")
    xgb_summary = summary[summary.Model == "XGBoost"].copy()
    xgb_folds.to_csv(tables / "04_XGB_Repeated_Random_5x10_CV.csv", index=False)
    xgb_summary.to_csv(tables / "05_XGB_Repeated_Random_5x10_Summary.csv", index=False)
    return folds, summary


def run_phase1(df, dirs, geometry):
    tables = dirs["Tables"]
    figures = dirs["Figures"]
    tuples, unique, gid, sizes, eligible, scaler, uz, nn, q50, q75 = geometry
    audit = dataset_audit(df, unique, sizes, eligible)
    audit.to_csv(tables / "01_Dataset_Audit.csv", index=False)
    thresholds = [("Exact_only_d0",0.0),("Q50_NN_distance",q50),("Q75_NN_distance",q75)]
    threshold_df = pd.DataFrame(thresholds, columns=["Threshold","Distance"])
    threshold_df.to_csv(tables / "06_Frozen_Distance_Thresholds.csv", index=False)
    nn_df = pd.DataFrame({"Composition_Group":np.arange(len(unique)),"Nearest_Neighbour_Distance":nn})
    nn_df.to_csv(tables / "07_Unique_Composition_NN_Distance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.hist(nn,bins=22)
    ax.axvline(q50,ls="--",label=f"Q50={q50:.3f}")
    ax.axvline(q75,ls=":",label=f"Q75={q75:.3f}")
    ax.set_xlabel("Nearest-neighbour chemistry distance")
    ax.set_ylabel("Unique compositions")
    ax.set_title("Frozen chemistry-distance thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "01_Frozen_Chemistry_Distance_Thresholds.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    folds, baseline_summary = run_baseline_robustness(df, tables)
    x = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    all_idx = np.arange(len(df))
    pair_rows = []
    for i in np.flatnonzero(eligible):
        g = gid[i]
        tr_present = all_idx[all_idx != i]
        tr_absent = all_idx[gid != g]
        m = xgb_model()
        m.fit(x[tr_present], y[tr_present])
        p_present = float(m.predict(x[[i]])[0])
        m = xgb_model()
        m.fit(x[tr_absent], y[tr_absent])
        p_absent = float(m.predict(x[[i]])[0])
        pair_rows.append((i,g,len(tr_present),len(tr_absent),y[i],p_present,p_absent,abs(y[i]-p_present),abs(y[i]-p_absent)))
    pair = pd.DataFrame(pair_rows, columns=["Row_Index","Composition_Group","Ntrain_Present","Ntrain_Absent","Actual_MPa","Pred_Present","Pred_Absent","AE_Present","AE_Absent"])
    pair["Delta_AE"] = pair.AE_Absent - pair.AE_Present
    pair.to_csv(tables / "08_Paired_Analogue_Record_Level.csv", index=False)
    global_rows = []
    for label, col in [("Analogue_present","Pred_Present"),("Analogue_absent","Pred_Absent")]:
        d = {"Condition":label}
        d.update(metric_dict(pair.Actual_MPa, pair[col]))
        global_rows.append(d)
    global_metrics = pd.DataFrame(global_rows)
    global_metrics.to_csv(tables / "09_Paired_Analogue_Global_Metrics.csv", index=False)
    pair_group = pair.groupby("Composition_Group").agg(N=("Row_Index","size"),MAE_Present=("AE_Present","mean"),MAE_Absent=("AE_Absent","mean"),Delta_AE=("Delta_AE","mean")).reset_index()
    pair_group.to_csv(tables / "10_Paired_Analogue_Group_Level.csv", index=False)
    ci = bootstrap_mean_ci(pair_group.Delta_AE.values)
    w = wilcoxon(pair_group.Delta_AE.values, alternative="two-sided", zero_method="wilcox")
    pair_stats = pd.DataFrame([{"N_records":len(pair),"N_groups":len(pair_group),"Record_mean_Delta_AE_MPa":pair.Delta_AE.mean(),"Record_median_Delta_AE_MPa":pair.Delta_AE.median(),"Fraction_records_error_increased":float((pair.Delta_AE>0).mean()),"Group_mean_Delta_AE_MPa":pair_group.Delta_AE.mean(),"Bootstrap95_Low":ci[0],"Bootstrap95_High":ci[1],"Wilcoxon_p":w.pvalue}])
    pair_stats.to_csv(tables / "11_Paired_Analogue_Statistics.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4.8))
    ax.boxplot([pair.AE_Present,pair.AE_Absent],tick_labels=["Analogue present","Analogue absent"],showfliers=False)
    ax.set_ylabel("Absolute error (MPa)")
    ax.set_title("Paired composition holdout: same 420 test records")
    fig.tight_layout()
    fig.savefig(figures / "02_Paired_Analogue_Error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.8))
    ax.hist(pair.Delta_AE,bins=30)
    ax.axvline(0,ls="--")
    ax.axvline(pair.Delta_AE.mean(),ls=":",label=f"Mean DeltaAE={pair.Delta_AE.mean():.2f} MPa")
    ax.set_xlabel("DeltaAE = absent - present (MPa)")
    ax.set_ylabel("Test records")
    ax.set_title("Penalty after removing identical-composition analogues")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "03_Analogue_Removal_Penalty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    record_z = scaler.transform(df[CHEMISTRY].to_numpy(float))
    dr = cdist(record_z, uz)
    sim_rows = []
    shap_rows = []
    for label, delta in thresholds:
        for i in range(len(df)):
            drec = dr[i, gid]
            tr = all_idx[(all_idx != i) & (drec > delta + 1e-12)]
            m = xgb_model()
            m.fit(x[tr], y[tr])
            pred = float(m.predict(x[[i]])[0])
            contrib = m.get_booster().predict(DMatrix(x[[i]], feature_names=FEATURES), pred_contribs=True)[0,:-1]
            retained = float(drec[tr].min()) if len(tr) else np.nan
            sim_rows.append((i,gid[i],label,delta,len(tr),len(all_idx)-1-len(tr),retained,y[i],pred,abs(y[i]-pred)))
            for f, v in zip(FEATURES, contrib):
                shap_rows.append((i,label,f,float(v),abs(float(v))))
    sim = pd.DataFrame(sim_rows, columns=["Row_Index","Composition_Group","Threshold","Distance","Ntrain","Nremoved_vs_LOO","Nearest_retained_distance","Actual_MPa","Predicted_MPa","AE_MPa"])
    sim.to_csv(tables / "12_Similarity_Controlled_Record_Level.csv", index=False)
    shap_values = pd.DataFrame(shap_rows, columns=["Row_Index","Threshold","Feature","SHAP","Abs_SHAP"])
    shap_values.to_csv(tables / "13_Similarity_Controlled_SHAP.csv", index=False)
    sim_summary_rows = []
    for label, delta in thresholds:
        s = sim[sim.Threshold == label]
        d = {"Threshold":label,"Distance":delta,"Mean_Ntrain":s.Ntrain.mean(),"Mean_Removed_vs_LOO":s.Nremoved_vs_LOO.mean(),"Composition_Macro_MAE":s.groupby("Composition_Group").AE_MPa.mean().mean()}
        d.update(metric_dict(s.Actual_MPa, s.Predicted_MPa))
        sim_summary_rows.append(d)
    sim_summary = pd.DataFrame(sim_summary_rows)
    sim_summary.to_csv(tables / "14_Similarity_Controlled_Summary.csv", index=False)
    shap_rank = shap_values.groupby(["Threshold","Feature"]).agg(MeanAbsSHAP=("Abs_SHAP","mean")).reset_index()
    shap_rank["Rank"] = shap_rank.groupby("Threshold").MeanAbsSHAP.rank(ascending=False,method="min")
    shap_rank.to_csv(tables / "15_SHAP_Rank_by_Threshold.csv", index=False)
    rank_matrix = shap_rank.pivot(index="Feature",columns="Threshold",values="Rank")
    rank_corr = rank_matrix.corr(method="spearman")
    rank_corr.to_csv(tables / "16_SHAP_Rank_Correlation.csv")
    fig, ax = plt.subplots(figsize=(7.5,5))
    ax.plot(sim_summary.Distance,sim_summary.MAE_MPa,marker="o",label="Record-weighted MAE")
    ax.plot(sim_summary.Distance,sim_summary.Composition_Macro_MAE,marker="s",ls="--",label="Composition-macro MAE")
    ax.set_xlabel("Chemistry exclusion distance")
    ax.set_ylabel("MAE (MPa)")
    ax.set_title("Transferability as chemical similarity is removed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "04_Transferability_Curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    exact_top = shap_rank[shap_rank.Threshold == "Exact_only_d0"].sort_values("Rank").head(10).Feature.tolist()
    ordered = [x[0] for x in thresholds]
    distance_values = [x[1] for x in thresholds]
    fig, ax = plt.subplots(figsize=(8,5.5))
    for f in exact_top:
        ranks = [float(shap_rank[(shap_rank.Threshold == lab) & (shap_rank.Feature == f)].Rank.iloc[0]) for lab in ordered]
        ax.plot(distance_values,ranks,marker="o",label=f)
    ax.invert_yaxis()
    ax.set_xlabel("Chemistry exclusion distance")
    ax.set_ylabel("SHAP importance rank")
    ax.set_title("Feature-rank stability across increasing chemical separation")
    ax.legend(ncol=2,fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "05_SHAP_Rank_Stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"audit":audit,"folds":folds,"baseline_summary":baseline_summary,"thresholds":threshold_df,"nn":nn_df,"pair":pair,"pair_global":global_metrics,"pair_group":pair_group,"pair_stats":pair_stats,"sim":sim,"sim_summary":sim_summary,"shap_rank":shap_rank,"rank_corr":rank_corr,"q50":q50,"q75":q75,"scaler":scaler,"uz":uz,"gid":gid,"unique":unique}


def run_size_matched_control(df, phase1, dirs, nrep):
    tables = dirs["Tables"]
    figures = dirs["Figures"]
    x = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    all_idx = np.arange(len(df))
    gid = phase1["gid"]
    scaler = phase1["scaler"]
    uz = phase1["uz"]
    dr = cdist(scaler.transform(df[CHEMISTRY].to_numpy(float)), uz)
    thresholds = {"Q50_NN_distance":phase1["q50"],"Q75_NN_distance":phase1["q75"]}
    sim = phase1["sim"]
    rows = []
    summary = []
    for label, delta in thresholds.items():
        print(f"Size-matched control {label}")
        for rep in range(nrep):
            preds = np.empty(len(df), float)
            ntrain = np.empty(len(df), int)
            for i in range(len(df)):
                drec = dr[i, gid]
                base = all_idx[(all_idx != i) & (drec > 1e-12)]
                target_n = int(((all_idx != i) & (drec > delta + 1e-12)).sum())
                rng = np.random.default_rng(SEED + (100000 if label.startswith("Q50") else 200000) + rep*1000 + i)
                tr = base if target_n >= len(base) else np.sort(rng.choice(base,size=target_n,replace=False))
                m = xgb_model()
                m.fit(x[tr], y[tr])
                preds[i] = m.predict(x[[i]])[0]
                ntrain[i] = len(tr)
            ae = np.abs(y-preds)
            for i in range(len(df)):
                rows.append((label,rep,i,gid[i],ntrain[i],y[i],preds[i],ae[i]))
            m = metric_dict(y,preds)
            m.update({"Threshold":label,"Repeat":rep,"Mean_Ntrain":ntrain.mean(),"Composition_Macro_MAE":pd.DataFrame({"g":gid,"ae":ae}).groupby("g").ae.mean().mean()})
            summary.append(m)
            print(f"{label} repeat {rep+1}/{nrep} MAE {m['MAE_MPa']:.4f}")
    record = pd.DataFrame(rows,columns=["Threshold","Repeat","Row_Index","Composition_Group","Ntrain","Actual_MPa","Predicted_MPa","AE_MPa"])
    record.to_csv(tables / "17_SizeMatched_Random_Control_Record_Level.csv", index=False)
    repeat_summary = pd.DataFrame(summary)
    repeat_summary.to_csv(tables / "18_SizeMatched_Random_Control_Repeat_Summary.csv", index=False)
    mean_record = record.groupby(["Threshold","Row_Index","Composition_Group"]).agg(Random_Mean_AE=("AE_MPa","mean"),Random_SD_AE=("AE_MPa","std")).reset_index()
    chem = sim[sim.Threshold.isin(thresholds)][["Threshold","Row_Index","Composition_Group","AE_MPa"]].rename(columns={"AE_MPa":"Chemical_AE"})
    merged = chem.merge(mean_record,on=["Threshold","Row_Index","Composition_Group"])
    merged["Delta_ChemMinusRandom"] = merged.Chemical_AE - merged.Random_Mean_AE
    merged.to_csv(tables / "19_Chemical_vs_SizeMatched_Record_Level.csv", index=False)
    stats = []
    for label in thresholds:
        z = merged[merged.Threshold == label]
        group = z.groupby("Composition_Group").Delta_ChemMinusRandom.mean()
        ci = bootstrap_mean_ci(group.values, seed=SEED+99)
        w = wilcoxon(group.values, alternative="two-sided", zero_method="wilcox")
        chem_mae = float(sim[sim.Threshold == label].AE_MPa.mean())
        random_values = repeat_summary[repeat_summary.Threshold == label].MAE_MPa
        stats.append({"Threshold":label,"Chemical_MAE_MPa":chem_mae,"Random_SizeMatched_MAE_Mean":random_values.mean(),"Random_SizeMatched_MAE_SD":random_values.std(),"Chemical_minus_Random_MAE":chem_mae-random_values.mean(),"GroupMean_DeltaAE":group.mean(),"Bootstrap95_Low":ci[0],"Bootstrap95_High":ci[1],"Wilcoxon_p":w.pvalue,"N_Repeats":nrep})
    stats = pd.DataFrame(stats)
    stats.to_csv(tables / "20_Chemical_vs_SizeMatched_Statistics.csv", index=False)
    precision = stats[["Threshold","Random_SizeMatched_MAE_Mean","Random_SizeMatched_MAE_SD","N_Repeats"]].copy()
    precision["Monte_Carlo_SE_MPa"] = precision.Random_SizeMatched_MAE_SD / np.sqrt(precision.N_Repeats)
    precision["Approx_95pct_MC_halfwidth_MPa"] = 1.96 * precision.Monte_Carlo_SE_MPa
    precision.to_csv(tables / "21_SizeMatched_MonteCarlo_Precision.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.7,5.2))
    ssum = phase1["sim_summary"]
    ax.plot(ssum.Distance,ssum.MAE_MPa,marker="o",lw=2,label="Chemistry exclusion")
    xs = [phase1["q50"],phase1["q75"]]
    ys = [repeat_summary[repeat_summary.Threshold == "Q50_NN_distance"].MAE_MPa.mean(),repeat_summary[repeat_summary.Threshold == "Q75_NN_distance"].MAE_MPa.mean()]
    es = [repeat_summary[repeat_summary.Threshold == "Q50_NN_distance"].MAE_MPa.std(),repeat_summary[repeat_summary.Threshold == "Q75_NN_distance"].MAE_MPa.std()]
    ax.errorbar(xs,ys,yerr=es,marker="s",ls="--",capsize=4,label=f"Size-matched random removal ({nrep} repeats)")
    ax.set_xlabel("Chemistry exclusion distance")
    ax.set_ylabel("MAE (MPa)")
    ax.set_title("Chemical similarity loss versus training-size loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "06_Chemical_vs_SizeMatched_Control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"record":record,"repeat_summary":repeat_summary,"merged":merged,"stats":stats,"precision":precision}


def run_within_route(df, phase1, dirs):
    tables = dirs["Tables"]
    figures = dirs["Figures"]
    work = df.copy()
    work["Process_Route"] = process_routes(work)
    route = work.loc[work.Process_Route == "TH1_C0_D0_T1"].copy().reset_index(drop=False)
    x = route[FEATURES].to_numpy(float)
    y = route[TARGET].to_numpy(float)
    tuples = [tuple(r) for r in route[CHEMISTRY].to_numpy(float)]
    unique = list(dict.fromkeys(tuples))
    mapping = {u:i for i,u in enumerate(unique)}
    gid = np.array([mapping[t] for t in tuples], int)
    sizes = np.bincount(gid,minlength=len(unique))
    eligible = sizes[gid] > 1
    all_idx = np.arange(len(route))
    pair_rows = []
    for i in np.flatnonzero(eligible):
        g = gid[i]
        tr_present = all_idx[all_idx != i]
        tr_absent = all_idx[gid != g]
        m = xgb_model()
        m.fit(x[tr_present],y[tr_present])
        p_present = float(m.predict(x[[i]])[0])
        m = xgb_model()
        m.fit(x[tr_absent],y[tr_absent])
        p_absent = float(m.predict(x[[i]])[0])
        pair_rows.append([i,g,y[i],p_present,p_absent,abs(y[i]-p_present),abs(y[i]-p_absent)])
    pair = pd.DataFrame(pair_rows,columns=["Row_Index","Composition_Group","Actual_MPa","Pred_Present","Pred_Absent","AE_Present","AE_Absent"])
    pair["Delta_AE"] = pair.AE_Absent - pair.AE_Present
    pair.to_csv(tables / "22_WithinRoute_Paired_Record_Level.csv", index=False)
    global_rows = []
    for label, col in [("Analogue_present","Pred_Present"),("Analogue_absent","Pred_Absent")]:
        d = {"Condition":label}
        d.update(metric_dict(pair.Actual_MPa,pair[col]))
        global_rows.append(d)
    global_metrics = pd.DataFrame(global_rows)
    global_metrics.to_csv(tables / "23_WithinRoute_Paired_Global_Metrics.csv", index=False)
    group = pair.groupby("Composition_Group").agg(N=("Row_Index","size"),MAE_Present=("AE_Present","mean"),MAE_Absent=("AE_Absent","mean"),Delta_AE=("Delta_AE","mean")).reset_index()
    group.to_csv(tables / "24_WithinRoute_Paired_Group_Level.csv", index=False)
    ci = bootstrap_mean_ci(group.Delta_AE.values)
    w = wilcoxon(group.Delta_AE.values,zero_method="wilcox")
    stats = pd.DataFrame([{"N_route_records":len(route),"N_paired_records":len(pair),"N_groups":len(group),"Group_mean_Delta_AE_MPa":group.Delta_AE.mean(),"Bootstrap95_Low":ci[0],"Bootstrap95_High":ci[1],"Wilcoxon_p":w.pvalue,"Fraction_records_error_increased":float((pair.Delta_AE>0).mean())}])
    stats.to_csv(tables / "25_WithinRoute_Paired_Statistics.csv", index=False)
    scaler = phase1["scaler"]
    route_z = scaler.transform(route[CHEMISTRY].to_numpy(float))
    unique_route_z = scaler.transform(np.asarray(unique,float))
    dr = cdist(route_z,unique_route_z)
    thresholds = [("Exact_only_d0",0.0),("Q50_NN_distance",phase1["q50"]),("Q75_NN_distance",phase1["q75"])]
    summary_rows = []
    for label, delta in thresholds:
        preds = []
        ntrain = []
        for i in range(len(route)):
            drec = dr[i,gid]
            tr = all_idx[(all_idx != i) & (drec > delta + 1e-12)]
            m = xgb_model()
            m.fit(x[tr],y[tr])
            preds.append(float(m.predict(x[[i]])[0]))
            ntrain.append(len(tr))
        d = {"Threshold":label,"Distance":delta,"Mean_Ntrain":float(np.mean(ntrain))}
        d.update(metric_dict(y,np.asarray(preds)))
        summary_rows.append(d)
    similarity = pd.DataFrame(summary_rows)
    similarity.to_csv(tables / "26_WithinRoute_Similarity_Summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7,4.8))
    ax.boxplot([pair.AE_Present,pair.AE_Absent],tick_labels=["Analogue present","Analogue absent"],showfliers=False)
    ax.set_ylabel("Absolute error (MPa)")
    ax.set_title("Within-route paired analogue test")
    fig.tight_layout()
    fig.savefig(figures / "07_WithinRoute_Paired_Error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.8))
    ax.plot(similarity.Distance,similarity.MAE_MPa,marker="o")
    ax.set_xlabel("Chemistry exclusion distance")
    ax.set_ylabel("MAE (MPa)")
    ax.set_title("Within-route transferability curve")
    fig.tight_layout()
    fig.savefig(figures / "08_WithinRoute_Transferability_Curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"route":route,"pair":pair,"global":global_metrics,"group":group,"stats":stats,"similarity":similarity}


def run_process_route_out(df, dirs):
    tables = dirs["Tables"]
    figures = dirs["Figures"]
    work = df.copy()
    work["Process_Route"] = process_routes(work)
    rows = []
    for route_name, n_test in work.Process_Route.value_counts().items():
        test = work.Process_Route == route_name
        train = ~test
        m = xgb_model()
        m.fit(work.loc[train,FEATURES].to_numpy(float),work.loc[train,TARGET].to_numpy(float))
        pred = m.predict(work.loc[test,FEATURES].to_numpy(float))
        d = {"Held_Out_Route":route_name,"N_Test":int(n_test)}
        d.update(metric_dict(work.loc[test,TARGET].to_numpy(float),pred))
        rows.append(d)
    route_out = pd.DataFrame(rows).sort_values("MAE_MPa",ascending=False).reset_index(drop=True)
    route_out.to_csv(tables / "27_Leave_One_Process_Route_Out_Stress_Test.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4,4.8))
    bars = ax.bar(route_out.Held_Out_Route,route_out.MAE_MPa)
    ax.set_ylabel("MAE (MPa)")
    ax.set_xlabel("Held-out process route")
    ax.set_title("Leave-one-process-route-out stress test")
    ax.tick_params(axis="x",rotation=18)
    for b, n in zip(bars,route_out.N_Test):
        ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"n={int(n)}",ha="center",va="bottom",fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "09_Process_Route_Out_Stress_Test.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return route_out


def run_feature_ablation(df, dirs):
    tables = dirs["Tables"]
    feature_sets = {"Composition_Only_9":CHEMISTRY,"Heat_Treatment_Only_12":HEAT,"Standard_21":HEAT+CHEMISTRY,"Complete_25":FEATURES}
    splitter = KFold(n_splits=5,shuffle=True,random_state=SEED)
    rows = []
    for name, cols in feature_sets.items():
        x = df[cols].to_numpy(float)
        y = df[TARGET].to_numpy(float)
        for fold, (tr,te) in enumerate(splitter.split(x),1):
            m = xgb_model()
            m.fit(x[tr],y[tr])
            pred = m.predict(x[te])
            rows.append({"Feature_Set":name,"Number_of_Features":len(cols),"Fold":fold,**metric_dict(y[te],pred)})
    folds = pd.DataFrame(rows)
    summary = folds.groupby(["Feature_Set","Number_of_Features"]).agg(R2_Mean=("R2","mean"),R2_SD=("R2","std"),MAE_Mean_MPa=("MAE_MPa","mean"),MAE_SD_MPa=("MAE_MPa","std"),RMSE_Mean_MPa=("RMSE_MPa","mean"),RMSE_SD_MPa=("RMSE_MPa","std")).reset_index()
    folds.to_csv(tables / "28_Feature_Set_Ablation_Folds.csv", index=False)
    summary.to_csv(tables / "29_Feature_Set_Ablation_Summary.csv", index=False)
    return folds, summary


def create_publication_figures(phase1, size_control, within_route, route_out, dirs):
    figures = dirs["Figures"]
    pub = dirs["Supplementary"] / "Publication_Figures"
    pub.mkdir(parents=True,exist_ok=True)
    pg = phase1["pair_global"].set_index("Condition")
    rg = within_route["global"].set_index("Condition")
    labels = ["Full benchmark","Dominant route"]
    present = [pg.loc["Analogue_present","MAE_MPa"],rg.loc["Analogue_present","MAE_MPa"]]
    absent = [pg.loc["Analogue_absent","MAE_MPa"],rg.loc["Analogue_absent","MAE_MPa"]]
    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    ax.bar(x-w/2,present,width=w,label="Analogue present")
    ax.bar(x+w/2,absent,width=w,label="Analogue absent")
    ax.set_xticks(x,labels)
    ax.set_ylabel("MAE (MPa)")
    ax.set_title("Prediction error after identical-composition analogue removal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pub / "Main_Figure_Paired_Analogue_Effect.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    ax.plot(phase1["sim_summary"].Distance,phase1["sim_summary"].MAE_MPa,marker="o",label="Full benchmark")
    ax.plot(within_route["similarity"].Distance,within_route["similarity"].MAE_MPa,marker="s",label="Dominant route")
    ax.set_xlabel("Chemistry exclusion distance")
    ax.set_ylabel("MAE (MPa)")
    ax.set_title("Error growth with increasing chemical separation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pub / "Main_Figure_Transferability_Full_vs_WithinRoute.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    stats = size_control["stats"]
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    x = np.arange(2)
    chem = stats.Chemical_MAE_MPa.to_numpy(float)
    rnd = stats.Random_SizeMatched_MAE_Mean.to_numpy(float)
    err = stats.Random_SizeMatched_MAE_SD.to_numpy(float)
    ax.bar(x-w/2,chem,width=w,label="Chemistry exclusion")
    ax.bar(x+w/2,rnd,width=w,yerr=err,capsize=4,label="Size-matched random removal")
    ax.set_xticks(x,["Q50","Q75"])
    ax.set_ylabel("MAE (MPa)")
    ax.set_title("Chemical-similarity loss versus training-size loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pub / "Main_Figure_Chemistry_vs_SizeMatched_Control.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    shutil_targets = [(figures / "01_Frozen_Chemistry_Distance_Thresholds.png",pub / "Supplementary_Figure_S1_Chemistry_Distance_Thresholds.png"),(figures / "07_WithinRoute_Paired_Error.png",pub / "Supplementary_Figure_S2_WithinRoute_Paired_Analogue_Error.png"),(figures / "09_Process_Route_Out_Stress_Test.png",pub / "Supplementary_Figure_S3_Process_Route_Out_Stress_Test.png")]
    import shutil
    for src, dst in shutil_targets:
        shutil.copy2(src,dst)
    return pub


def create_supplementary_workbook(df, phase1, size_control, within_route, route_out, ablation, dirs):
    path = dirs["Supplementary"] / "Supplementary_Tables_S1-S6_Generated.xlsx"
    routes = process_routes(df)
    route_summary = pd.DataFrame({"Process_Route":routes}).value_counts().rename("N").reset_index()
    route_summary = route_summary.merge(df.assign(Process_Route=routes).groupby("Process_Route")[TARGET].agg(["mean","std","min","max"]).reset_index(),on="Process_Route")
    descriptive = df[FEATURES+[TARGET]].describe().T.reset_index().rename(columns={"index":"Variable"})
    composition_table = pd.DataFrame(phase1["unique"],columns=CHEMISTRY)
    composition_table.insert(0,"Composition_Group",np.arange(len(composition_table)))
    counts = pd.Series(phase1["gid"]).value_counts().sort_index()
    composition_table["N_records"] = composition_table.Composition_Group.map(counts)
    with pd.ExcelWriter(path,engine="openpyxl") as writer:
        phase1["audit"].to_excel(writer,sheet_name="Table S1",index=False,startrow=0)
        route_summary.to_excel(writer,sheet_name="Table S1",index=False,startrow=10)
        descriptive.to_excel(writer,sheet_name="Table S1",index=False,startrow=18)
        composition_table.to_excel(writer,sheet_name="Table S1",index=False,startrow=50)
        phase1["folds"].to_excel(writer,sheet_name="Table S2",index=False,startrow=0)
        phase1["baseline_summary"].to_excel(writer,sheet_name="Table S2",index=False,startrow=len(phase1["folds"])+3)
        phase1["pair_global"].to_excel(writer,sheet_name="Table S3",index=False,startrow=0)
        phase1["pair_stats"].to_excel(writer,sheet_name="Table S3",index=False,startrow=6)
        phase1["pair"].to_excel(writer,sheet_name="Table S3",index=False,startrow=11)
        phase1["pair_group"].to_excel(writer,sheet_name="Table S3",index=False,startrow=11+len(phase1["pair"])+3)
        phase1["thresholds"].to_excel(writer,sheet_name="Table S4",index=False,startrow=0)
        phase1["sim_summary"].to_excel(writer,sheet_name="Table S4",index=False,startrow=6)
        phase1["nn"].to_excel(writer,sheet_name="Table S4",index=False,startrow=13)
        phase1["sim"].to_excel(writer,sheet_name="Table S4",index=False,startrow=13+len(phase1["nn"])+3)
        size_control["stats"].to_excel(writer,sheet_name="Table S5",index=False,startrow=0)
        size_control["precision"].to_excel(writer,sheet_name="Table S5",index=False,startrow=6)
        size_control["repeat_summary"].to_excel(writer,sheet_name="Table S5",index=False,startrow=12)
        phase1["shap_rank"].to_excel(writer,sheet_name="Table S6",index=False,startrow=0)
        phase1["rank_corr"].to_excel(writer,sheet_name="Table S6",startrow=len(phase1["shap_rank"])+3)
    return path


def write_summary(root, phase1, size_control, within_route, route_out, ablation, runtime, data_path, nrep):
    pg = phase1["pair_global"].set_index("Condition")
    ps = phase1["pair_stats"].iloc[0]
    rg = within_route["global"].set_index("Condition")
    rs = within_route["stats"].iloc[0]
    lines = ["NIMS FATIGUE REPRODUCIBILITY SUMMARY",f"Dataset: {data_path}",f"Records: 437",f"Predictors: 25",f"Unique exact compositions: {len(phase1['unique'])}",f"Repeated composition groups: {int((pd.Series(phase1['gid']).value_counts()>1).sum())}",f"Paired eligible records: {len(phase1['pair'])}",f"Q50: {phase1['q50']:.12f}",f"Q75: {phase1['q75']:.12f}",f"Analogue present MAE: {pg.loc['Analogue_present','MAE_MPa']:.6f}",f"Analogue absent MAE: {pg.loc['Analogue_absent','MAE_MPa']:.6f}",f"Group mean DeltaAE: {ps.Group_mean_Delta_AE_MPa:.6f}",f"Paired bootstrap 95 percent CI: {ps.Bootstrap95_Low:.6f} to {ps.Bootstrap95_High:.6f}",f"Paired Wilcoxon p: {ps.Wilcoxon_p:.12g}",f"Within-route analogue present MAE: {rg.loc['Analogue_present','MAE_MPa']:.6f}",f"Within-route analogue absent MAE: {rg.loc['Analogue_absent','MAE_MPa']:.6f}",f"Within-route group mean DeltaAE: {rs.Group_mean_Delta_AE_MPa:.6f}",f"Size-matched repeats: {nrep}",f"Runtime minutes: {runtime/60:.2f}"]
    for _, r in size_control["stats"].iterrows():
        lines.append(f"{r.Threshold} chemistry MAE: {r.Chemical_MAE_MPa:.6f}")
        lines.append(f"{r.Threshold} random size-matched MAE: {r.Random_SizeMatched_MAE_Mean:.6f}")
        lines.append(f"{r.Threshold} group mean chemistry-specific DeltaAE: {r.GroupMean_DeltaAE:.6f}")
        lines.append(f"{r.Threshold} Wilcoxon p: {r.Wilcoxon_p:.12g}")
    (root / "FINAL_RESULTS_SUMMARY.txt").write_text("\n".join(lines),encoding="utf-8")


def write_manifest(root, data_path, clean_path, nrep, runtime):
    manifest = {"data_path":str(data_path),"canonical_data_copy":str(clean_path),"seed":SEED,"size_matched_repeats":nrep,"xgboost_parameters":XGB_PARAMS,"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__,"xgboost":xgboost.__version__,"runtime_seconds":runtime}
    (root / "RUN_MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def main():
    global RF_ESTIMATORS, BASELINE_REPEATS, BOOTSTRAP_REPEATS
    start = time.time()
    args = parse_arguments()
    if args.quick_test:
        XGB_PARAMS["n_estimators"] = 30
        RF_ESTIMATORS = 30
        BASELINE_REPEATS = 1
        BOOTSTRAP_REPEATS = 200
        args.nrep_size_matched = 1
    data_path = resolve_data_path(args.data)
    output_root = Path(args.output).expanduser().resolve() if args.output else data_path.parent / "NIMS_Fatigue_Submission_Outputs"
    output_root.mkdir(parents=True,exist_ok=True)
    dirs = ensure_dirs(output_root)
    print(f"Dataset: {data_path}")
    print(f"Output: {output_root}")
    df, sheet, clean_path = load_dataset(data_path, output_root)
    print(f"Sheet: {sheet}")
    print(f"Shape: {df.shape}")
    geometry = composition_geometry(df)
    print("Running Phase 1")
    phase1 = run_phase1(df, dirs, geometry)
    print("Running size-matched control")
    size_control = run_size_matched_control(df, phase1, dirs, args.nrep_size_matched)
    print("Running within-route robustness")
    within_route = run_within_route(df, phase1, dirs)
    print("Running process-route-out stress test")
    route_out = run_process_route_out(df, dirs)
    print("Running feature-set ablation")
    ablation = run_feature_ablation(df, dirs)
    print("Saving final inference model")
    save_final_model(df, output_root)
    print("Generating publication figures")
    publication_dir = create_publication_figures(phase1,size_control,within_route,route_out,dirs)
    print("Generating supplementary workbook")
    workbook = create_supplementary_workbook(df,phase1,size_control,within_route,route_out,ablation,dirs)
    runtime = time.time() - start
    write_summary(output_root,phase1,size_control,within_route,route_out,ablation,runtime,data_path,args.nrep_size_matched)
    write_manifest(output_root,data_path,clean_path,args.nrep_size_matched,runtime)
    print("Analysis completed successfully")
    print(f"Publication figures: {publication_dir}")
    print(f"Supplementary workbook: {workbook}")
    print(f"Final summary: {output_root / 'FINAL_RESULTS_SUMMARY.txt'}")


if __name__ == "__main__":
    main()
