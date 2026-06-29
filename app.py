from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.core import compute_metrics
from src.visualizations import *
st.set_page_config(page_title="InsureRisk | EXL Insurance Risk", layout="wide", page_icon="\U0001f3e6")
with st.sidebar:
    st.header("\u2699 Config"); n = st.slider("Samples", 2000, 20000, 10000, 1000)
    tau = st.slider("Threshold", 0.05, 0.95, 0.50, 0.05)
    st.caption("EXL | Solvency II | NAIC AI Bulletin")
data = make_synthetic(n=n); b = train_all_models(data)
y_test = b["y_test"]; y_probas = {n: b["results"][n]["y_proba"] for n in b["results"]}
best = max(b["results"], key=lambda n: b["results"][n]["metrics"].get("roc_auc", 0))
c1,c2,c3,c4 = st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Claim Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4 = st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f3af Risk Segmentation","\U0001f4b0 Pricing"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    col_a,col_b = st.columns(2)
    with col_a:
        fig,ax = plt.subplots(figsize=(5,3)); _style()
        ax.bar(["No Claim","Claim"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
        for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
        ax.set_title("Claim Distribution",color="white"); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        st.markdown("**Key risk factors:** prior_claims_3yr, credit_score, region_risk_score, vehicle_value, driving_experience, annual_mileage")
with t2:
    rows = [{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b = st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test, y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test, y_probas))
    st.pyplot(plot_confusion_matrix(y_test, b["results"]["XGBoost"]["y_pred"], "XGBoost"))
    cv = cross_validate(data)
    cvr = [{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Risk Segmentation & Loss Ratio Analysis")
    st.latex(r"\text{Loss Ratio} = \frac{\text{Incurred Losses}}{\text{Earned Premiums}}")
    xgb_y = b["results"]["XGBoost"]["y_proba"]
    segs = {"Low Risk": (xgb_y<.10).sum(), "Medium Risk": ((xgb_y>=.10)&(xgb_y<.25)).sum(), "High Risk": ((xgb_y>=.25)&(xgb_y<.40)).sum(), "Very High": (xgb_y>=.40).sum()}
    fig,ax = plt.subplots(figsize=(7,4)); _style()
    names = list(segs.keys()); vals = list(segs.values())
    colors = ["#22c55e","#fbbf24","#f97316","#f43f5e"]
    bars = ax.bar(names, vals, color=colors, alpha=0.7)
    ax.set_title("Policy Count by Risk Segment",color="white"); ax.grid(True,alpha=.2,axis="y")
    st.pyplot(fig)
    premium = st.number_input("Annual Premium ($)", 500, 5000, 1200, 100)
    rows2 = []
    for name,cnt in segs.items():
        avg_pd = {"Low Risk":.05,"Medium Risk":.175,"High Risk":.325,"Very High":.55}[name]
        exp_loss = avg_pd * premium * cnt
        loss_ratio = exp_loss / (premium * cnt) if cnt >0 else 0
        rows2.append({"Segment":name, "Count":cnt, "Avg PD":f"{avg_pd:.1%}", "Expected Loss":f"${exp_loss:,.0f}", "Loss Ratio":f"{loss_ratio:.1%}"})
    st.dataframe(pd.DataFrame(rows2),use_container_width=True,hide_index=True)
with t4:
    st.subheader("Risk-Based Premium Pricing")
    st.latex(r"\text{Premium}_i = \text{Base Rate} \times (1 + \text{Risk Loading}_i)")
    base = st.slider("Base Rate", 0.02, 0.10, 0.05, 0.005)
    risk_load = base * xgb_y
    prem = 1200 * (1 + risk_load * 10)
    fig,ax = plt.subplots(figsize=(8,4)); _style()
    bins = np.linspace(0,prem.max(),30)
    for label,mask,c in [("No Claim",y_test==0,"#22c55e"),("Claim",y_test==1,"#f43f5e")]:
        ax.hist(prem[mask],bins=bins,alpha=0.5,color=c,label=f"{label} (n={mask.sum()})",density=True)
    ax.set_xlabel("Annual Premium ($)"); ax.set_ylabel("Density")
    ax.set_title("Premium Distribution by Claim Outcome",color="white")
    ax.legend(fontsize=8); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.dataframe(pd.DataFrame({"Avg Premium No Claim": f"${prem[y_test==0].mean():.0f}", "Avg Premium Claim": f"${prem[y_test==1].mean():.0f}", "Risk Load Factor": f"{risk_load.mean():.4f}"}, index=[0]), use_container_width=True,hide_index=True)
    st.subheader("Actuarial Solvency Framework")
    st.latex(r"\psi(u) = \mathbb{P}\!\left(\inf_{t>0} U(t) < 0\right), \quad U(t) = u + ct - S(t)")
    st.caption("Ruin probability: insurer's surplus U(t) must stay above zero; premium rate c and claim process S(t) determine solvency thresholds.")
    st.latex(r"\text{SCR} = \text{VaR}_{99.5\%}(\Delta\text{NAV}) \approx \mu_{\text{NAV}} + 2.58\,\sigma_{\text{NAV}}")
    st.caption("Solvency II capital requirement: buffer needed to absorb 1-in-200-year losses in net asset value.")
    st.latex(r"f(x; k, \theta) = \frac{x^{k-1}e^{-x/\theta}}{\theta^k\Gamma(k)}, \quad x \ge 0")
    st.caption("Gamma-distributed claim severity: shape k controls skewness, scale θ captures heavy-tailed loss behavior for catastrophic risk.")
    st.latex(r"C\big(F_1(x_1), F_2(x_2), \ldots, F_n(x_n)\big) = \mathbb{P}(X_1 \le x_1, \ldots, X_n \le x_n)")
    st.caption("Copula dependence between policy risks (e.g., auto & home) beyond linear correlation. Gaussian or t-copula used for aggregate loss modeling.")
    st.caption("EXL's insurance analytics practice deploys these actuarial models for Solvency II compliance (Pillar 1 capital calculation), NAIC AI bulletin governance (model risk management), and pricing transformation for top-10 US carriers. EXL's proprietary RiskInsight platform integrates GLM, GBM, and copula models for end-to-end ratemaking.")
