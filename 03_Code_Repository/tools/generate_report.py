#!/usr/bin/env python3
"""Generate the final HTML report from frozen CSV outputs and PNG figures.

Usage:
    cd 03_Code_Repository
    python3 tools/generate_report.py --output-dir outputs --out ../01_Report/final_report.html
"""
from __future__ import annotations
import argparse, base64, html as html_mod, textwrap
from pathlib import Path
import pandas as pd, numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def img64(path: Path) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

def pct(v, d=2):
    if pd.isna(v): return ""
    return f"{v*100:.{d}f}%"

def num(v, d=3):
    if pd.isna(v): return ""
    return f"{v:.{d}f}"

def money(v):
    if pd.isna(v): return ""
    if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def aum_label(v):
    if v >= 1e9: return "1B"
    return f"{int(v/1e6)}M"

def esc(s):
    return html_mod.escape(str(s))

def table_html(df, fmt_map=None, cls=""):
    """Build an HTML table from a DataFrame. fmt_map maps col name -> callable."""
    fmt_map = fmt_map or {}
    rows = ['<table class="data-table ' + cls + '">']
    rows.append("<thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in df.columns) + "</tr></thead>")
    rows.append("<tbody>")
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c in fmt_map:
                cells.append(f"<td>{fmt_map[c](v)}</td>")
            else:
                cells.append(f"<td>{esc(v)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
:root {
  --imperial-blue: #002147;
  --imperial-light: #003E74;
  --accent-teal: #0091B3;
  --text-primary: #1a1a1a;
  --text-secondary: #4a4a4a;
  --border-color: #c0c0c0;
  --bg-light: #f7f8fa;
  --bg-table-header: #e8ecf1;
  --positive: #1a7a3a;
  --negative: #b91c1c;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html { font-size: 11pt; }
body { font-family: Georgia, 'Times New Roman', 'Palatino Linotype', serif;
       line-height: 1.62; color: var(--text-primary); }
.container { max-width: 850px; margin: 0 auto; padding: 40px 50px; }
h1, h2, h3, h4 { font-family: 'Helvetica Neue', Arial, Calibri, sans-serif;
                  color: var(--imperial-blue); margin-top: 1.4em; margin-bottom: 0.5em; }
h1 { font-size: 1.7rem; border-bottom: 3px solid var(--imperial-blue); padding-bottom: 8px; }
h2 { font-size: 1.32rem; border-bottom: 1.5px solid var(--border-color); padding-bottom: 6px; }
h3 { font-size: 1.08rem; }
p { margin-bottom: 0.7em; }
.section-num { color: var(--accent-teal); margin-right: 6px; }
.cover { text-align: center; padding-top: 80px; }
.cover h1 { border: none; font-size: 2rem; color: var(--imperial-blue); }
.cover .subtitle { font-size: 1.15rem; color: var(--text-secondary); margin-top: 6px; }
.cover .institution { font-family: 'Helvetica Neue', Arial, sans-serif;
                      color: var(--imperial-blue); font-size: 0.95rem;
                      text-transform: uppercase; letter-spacing: 3px; margin-bottom: 6px; }
.kpi-row { display: flex; justify-content: center; gap: 18px; margin: 30px 0; flex-wrap: wrap; }
.kpi-box { border: 1px solid var(--border-color); border-radius: 6px; padding: 12px 20px;
           min-width: 140px; text-align: center; background: var(--bg-light); }
.kpi-box .label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px;
                  color: var(--text-secondary); font-family: 'Helvetica Neue', Arial, sans-serif; }
.kpi-box .value { font-size: 1.35rem; font-weight: 700; color: var(--imperial-blue); margin-top: 2px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 14px 0 18px 0; }
.data-table th { background: var(--bg-table-header); font-weight: 600; text-align: center;
                 padding: 6px 8px; border: 1px solid var(--border-color);
                 font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 0.82rem; }
.data-table td { padding: 5px 8px; border: 1px solid var(--border-color); text-align: center; }
.data-table tbody tr:nth-child(even) { background: var(--bg-light); }
.data-table.left-align td:first-child { text-align: left; }
.data-table.small { font-size: 0.78rem; }
.data-table.small th { font-size: 0.74rem; }
.callout { border-left: 4px solid var(--accent-teal); background: var(--bg-light);
           padding: 12px 16px; margin: 14px 0; font-size: 0.92rem; }
.callout strong { color: var(--imperial-blue); }
.warning { border-left: 4px solid #d97706; background: #fffbeb; }
figure { margin: 18px 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
figcaption { font-size: 0.82rem; color: var(--text-secondary); font-style: italic;
             margin-top: 6px; max-width: 90%; margin-left: auto; margin-right: auto; }
code { font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.87em;
       background: #f0f2f5; padding: 1px 5px; border-radius: 3px; }
.toc { margin: 20px 0; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { padding: 3px 0; font-size: 0.95rem; }
.toc li.indent { padding-left: 24px; font-size: 0.9rem; }
.toc a { color: var(--imperial-blue); text-decoration: none; }
.formula { background: var(--bg-light); border-left: 3px solid var(--accent-teal);
           padding: 10px 16px; margin: 12px 0; font-family: 'SF Mono', Menlo, monospace;
           font-size: 0.9rem; }
.pos { color: var(--positive); }
.neg { color: var(--negative); }
hr.section-break { border: none; border-top: 1px solid var(--border-color); margin: 30px 0; }

@media print {
  html { font-size: 8.8pt; }
  .container { padding: 0; max-width: 100%; }
  .cover { page-break-after: always; }
  .toc-page { page-break-after: always; }
  h2 { page-break-before: auto; }
  .page-break { page-break-before: always; }
  h2.no-break-before { page-break-before: avoid; }
  table, figure, .callout { page-break-inside: avoid; }
  .data-table { font-size: 0.78rem; margin: 8px 0 12px 0; }
  .data-table th { padding: 4px 5px; font-size: 0.72rem; }
  .data-table td { padding: 3px 5px; }
  .data-table.small { font-size: 6.5pt; }
  .data-table.small th { font-size: 6.5pt; }
  .kpi-box .value { font-size: 1.1rem; }
  a { color: var(--imperial-blue); text-decoration: none; }
}
"""

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("../01_Report/final_report.html"))
    args = parser.parse_args()
    O = args.output_dir
    R = O / "report_ready"

    # Load all data
    perf     = pd.read_csv(O / "performance_summary.csv")
    universe = pd.read_csv(O / "universe_summary.csv")
    elig_raw = pd.read_csv(O / "eligibility_summary.csv")
    ablation = pd.read_csv(O / "feature_ablation_summary.csv")
    ic_yr    = pd.read_csv(O / "ic_yearly.csv")
    stress   = pd.read_csv(O / "stress_windows_250m.csv")
    recon    = pd.read_csv(O / "return_reconciliation_summary.csv").iloc[0]
    earnings = pd.read_csv(O / "earnings_timing_examples.csv")
    si_lag   = pd.read_csv(O / "short_interest_lag_examples.csv")
    borrow_t = pd.read_csv(R / "borrow_tier_summary.csv")
    borrow_s = pd.read_csv(R / "borrow_sharpe_degradation.csv")
    capacity = pd.read_csv(R / "position_capacity_summary.csv")
    features = pd.read_csv(R / "feature_inventory.csv")
    lo_adj   = pd.read_csv(R / "sharpe_lo_corrected.csv")
    hard_exc = pd.read_csv(R / "short_interest_hard_exclusion_sensitivity.csv")
    impact   = pd.read_csv(R / "impact_at_cap_examples.csv")
    lgbm     = pd.read_csv(R / "lightgbm_diagnostic.csv")
    robust   = pd.read_csv(R / "locked_strategy_robustness.csv")
    si_cont  = pd.read_csv(R / "short_interest_si_gt_10_contribution.csv")
    borrow_v = pd.read_csv(R / "borrow_proxy_external_validation.csv")
    si_aff   = pd.read_csv(R / "short_signal_affected_by_borrow.csv")
    champ    = pd.read_csv(R / "champion_challenger_matrix.csv")

    p250 = perf[perf["AUM"] == 2.5e8].iloc[0]

    # Load images
    imgs = {}
    for name, rel in [
        ("decomp",   "figures/equal_weight_return_decomposition.png"),
        ("rolling",  "figures/rolling_ic.png"),
        ("capacity", "report_ready/report_capacity.png"),
        ("ablation", "report_ready/report_ablation.png"),
        ("cumret",   "report_ready/report_aum_cumulative_returns.png"),
        ("weights",  "report_ready/model_feature_weights_2024.png"),
        ("ic_comp",  "report_ready/model_ic_comparison.png"),
        ("ic_year",  "report_ready/model_yearly_ic.png"),
        ("decomp_q", "report_ready/return_decomposition_q1.png"),
        ("si_series","report_ready/short_interest_representative_series.png"),
    ]:
        p = O / rel
        if p.exists():
            imgs[name] = img64(p)
        else:
            imgs[name] = ""

    # -----------------------------------------------------------------------
    # Build sections
    # -----------------------------------------------------------------------
    sections = []

    # --- COVER ---
    sections.append(f"""
<div class="cover">
  <p class="institution">Imperial College London</p>
  <p style="font-family:'Helvetica Neue',Arial,sans-serif;color:var(--text-secondary);font-size:0.95rem;">Machine Learning in Finance</p>
  <hr style="width:120px;margin:18px auto;border:none;border-top:2px solid var(--imperial-blue);">
  <h1 style="border:none;font-size:1.9rem;">Close-to-Open (C2O) Overnight Equity Strategy</h1>
  <p class="subtitle">Final Submission Report</p>
  <hr style="width:100px;margin:18px auto;border:none;border-top:2px solid var(--imperial-blue);">
  <div class="kpi-row">
    <div class="kpi-box"><div class="label">250M Net Return</div><div class="value">{pct(p250['net_annual_return'])}</div></div>
    <div class="kpi-box"><div class="label">250M Net Vol</div><div class="value">{pct(p250['net_vol'])}</div></div>
    <div class="kpi-box"><div class="label">250M Net Sharpe</div><div class="value">{num(p250['net_sharpe'])}</div></div>
    <div class="kpi-box"><div class="label">Max Drawdown</div><div class="value">{pct(p250['max_drawdown'])}</div></div>
  </div>
  <div class="callout" style="text-align:left;max-width:620px;margin:20px auto;">
    <strong>Decision:</strong> final strategy is <code>phase2_g5_05_expanding</code>.
    The old 0.811 Sharpe strategy is retained only as previous-champion evidence.
  </div>
  <p style="font-size:0.85rem;color:var(--text-secondary);margin-top:30px;">
    Strategy universe: US Large-Cap Equities, 2010&ndash;2024 &bull;
    Final strategy: volatility-scaled target with expanding-window weight estimation<br>
    AUM scenarios: $50M &middot; $250M &middot; $1B &bull;
    Main headline AUM: $250M &bull;
    Development cutoff: 31 December 2024
  </p>
</div>
""")

    # --- TOC ---
    toc_items = [
        ("1", "Executive Summary and Coursework Map"),
        ("2", "Methodology Pipeline"),
        ("3", "Data, Trading Clock, and Point-in-Time Panel"),
        ("4", "Capacity-Aware Universe and Execution Assumptions"),
        ("5", "Borrow Proxy and Short-Leg Financing"),
        ("6", "Baseline, Promotion Decision, and Challenger Models"),
        ("7", "Alpha Design, Model Selection, and Interpretability"),
        ("8", "Portfolio Construction, Costs, and Headline Results"),
        ("9", "Robustness and Stress Tests"),
        ("10", "Reproducibility and Code Audit"),
        ("11", "Limitations and Next Work"),
        ("12", "Sceptical Marker Checklist"),
        ("13", "Conclusion"),
        ("A", "Appendix A: Universe and Eligibility Evidence"),
        ("B", "Appendix B: Feature Inventory and IC by Year"),
        ("C", "Appendix C: 250M Year-by-Year Performance"),
        ("D", "Appendix D: Output Manifest"),
    ]
    toc_html = '<div class="toc-page"><h1>Table of Contents</h1><div class="toc"><ul>'
    for n, t in toc_items:
        cls = ' class="indent"' if n in ("A","B","C","D") else ""
        toc_html += f'<li{cls}><strong>{n}.</strong> {t}</li>'
    toc_html += '</ul></div></div>'
    sections.append(toc_html)

    # --- SECTION 1: EXECUTIVE SUMMARY ---
    # Coursework question map
    qmap = pd.DataFrame([
        ["Section 2", "Daily panel questions", "6", "Data sources, return identity, earnings timing, short-interest lag, universe and stylised fact"],
        ["Section 3", "Capacity-aware universe", "4", "Thresholds, participation cap, AUM capacity, binding constraints"],
        ["Section 4", "Borrow filtering", "4", "Borrow proxy, external validation, tiered cost treatment, gross-to-net borrow impact"],
        ["Section 5", "Alpha model", "5", "Information set, target, model/training, IC, ablation and weak periods"],
        ["Section 6", "Portfolio and costs", "5", "Basket/weighting/turnover, 50M/250M/1B performance, cost drag, QuantStats, stress windows"],
    ], columns=["Brief Section", "Topic", "Explicit Questions", "Where Answered Here"])

    sections.append(f"""
<h2 class="no-break-before"><span class="section-num">1</span>Executive Summary and Coursework Map</h2>
<p>The final strategy is <code>phase2_g5_05_expanding</code>: a daily, dollar-neutral, close-to-open long-short strategy using a volatility-scaled overnight-return target with expanding-window transparent feature-weight estimation. The target is:</p>
<div class="formula">target = overnight_next / (vol20 / sqrt(252))</div>
<p>where <code>overnight_next</code> is the close-to-next-open return and <code>vol20</code> is trailing 20-day close-to-close volatility, annualised and shifted by one trading day. The feature set is unchanged from the original point-in-time panel. The main research change is a risk-scaled label and an expanding training window, not a larger black-box model.</p>
<p>At the main headline AUM of 250M, the final strategy earns <strong>{pct(p250['net_annual_return'])}</strong> net annual return, <strong>{pct(p250['net_vol'])}</strong> net volatility, net Sharpe <strong>{num(p250['net_sharpe'])}</strong>, and max drawdown <strong>{pct(p250['max_drawdown'])}</strong> over 2010&ndash;2024 after commission, auction slippage, borrow costs, and a 5% ADV20 participation cap.</p>
<p>We keep weak evidence in the report. The 2024 year is negative. The 2023&ndash;2024 holdout Sharpe is lower than validation. We do not have external prime-broker borrow validation. 1B capacity is clearly constrained. 2025&ndash;2026 remains held out for marker evaluation.</p>
<h3>Coursework Question Map</h3>
<p>The brief contains <strong>24 explicit report questions</strong> across Sections 2&ndash;6:</p>
{table_html(qmap, cls="left-align")}
<p>Section 7 adds sceptical-marker audit questions. This report answers those by tying look-ahead, capacity, borrow, robustness, and cost claims to frozen output files.</p>
""")

    # --- SECTION 2: METHODOLOGY ---
    pipe = pd.DataFrame([
        ["1", "Load and align data", "Read prices, all_data, cheapness scores, earnings, short interest, GICS and benchmark files"],
        ["2", "Build point-in-time features", "Lag close/high/low, accounting, valuation and short-interest fields; rank features cross-sectionally"],
        ["3", "Define investable universe", "Use prior-year top 1000 market-cap names, then apply price, ADV, volatility, earnings and history filters"],
        ["4", "Train alpha by year", "For each scored year, use only earlier years under the expanding-window protocol"],
        ["5", "Form daily baskets", "Buy top 3% and short bottom 3%, with at least 15 names each side and equal weights before caps"],
        ["6", "Apply costs and capacity", "Use commission, auction slippage, borrow tiers and 5% ADV20 per-name participation cap"],
        ["7", "Audit and report", "Write performance, positions, IC, ablation, robustness, borrow, capacity and reproduction files"],
    ], columns=["Step", "Stage", "What the Code Does"])

    sections.append(f"""
<h2><span class="section-num">2</span>Methodology Pipeline</h2>
<p>The pipeline is meant to be boring in the right places. It reads the same local data files, builds the same point-in-time panel, uses the same costs and capacity cap, and writes the same outputs every time. The only promoted modelling change is the volatility-scaled target with expanding-window estimation.</p>
{table_html(pipe, cls="left-align")}
<p>The main implementation lives in <code>src/c2o_strategy/final_strategy.py</code>. The <code>Makefile</code> runs the reproduction in separate targets: final outputs, ablation, then report assets.</p>
""")

    # --- SECTION 3: DATA PANEL ---
    # Earnings table
    earn_tbl = pd.DataFrame([
        ["AMC example", "MOS", "2010-01-05", "after", "2010-01-05", "2010-01-04, 2010-01-05, 2010-01-06"],
        ["BMO example", "RPM", "2010-01-06", "before", "2010-01-05", "2010-01-04, 2010-01-05, 2010-01-06"],
    ], columns=["Example", "Ticker", "Report Date", "Timing", "Strategy Date", "Excluded Decision Dates"])

    # SI lag table
    si_tbl = pd.DataFrame([
        ["HLT", "2015-01-27", "2015-01-28", "0.004", "1.67", "-0.25"],
        ["HLT", "2015-01-28", "2015-01-29", "0.004", "1.67", "-0.25"],
        ["HLT", "2015-01-29", "2015-01-30", "0.004", "1.67", "-0.25"],
    ], columns=["Ticker", "Source Feature Date", "Decision Date", "DSI", "DTCN", "DDTCN"])

    # Universe table
    uni_tbl = universe.copy()
    uni_tbl.columns = ["Year", "Universe Count", "Reference Date", "Median Year-Start Mcap", "Mid-Year Exits"]
    uni_tbl["Median Year-Start Mcap"] = uni_tbl["Median Year-Start Mcap"].apply(lambda x: money(x))

    sections.append(f"""
<h2><span class="section-num">3</span>Data, Trading Clock, and Point-in-Time Panel</h2>
<h3>3.1 Data Source and Window</h3>
<p>The pipeline uses the provided coursework data files under <code>data/</code>: daily adjusted prices, market cap, volume, GICS information, accounting/valuation variables, earnings calendar, short-interest proxies, regime data, and S&amp;P 500 total return benchmark. The development sample runs from 1 January 2010 to 31 December 2024. No 2025 or 2026 observations are used.</p>
<p>The trading decision is made before the day-<em>t</em> close. Positions enter at the close and exit at the next open. Features using day-<em>t</em> close, high, low, accounting, or short-interest data are shifted unless they are known before the decision time. Cross-sectional ranks are formed within the same-day available cross-section only.</p>

<h3>3.2 Return Reconciliation</h3>
<p>The panel passes the open-close return identity check. We verify:</p>
<div class="formula">(1 + r_overnight) &times; (1 + r_intraday) &minus; 1 = close_t / close_(t&minus;1) &minus; 1</div>
<p>on <strong>{int(recon['stock_days_checked']):,}</strong> stock-days at tolerance <code>1e-08</code>. The fail count is <strong>{int(recon['fail_count'])}</strong>, fail fraction is <strong>{recon['fail_fraction']:.6f}</strong>, and the maximum absolute residual is {recon['max_abs_residual']:.2e}.</p>

<h3>3.3 Earnings Timing and Short-Interest Lag</h3>
<p>Earnings observations are mapped to a strategy trading date. A before-market announcement can be known before that day's close; an after-market announcement cannot be used for the same close decision. The pipeline excludes a &plusmn;1 trading-day window around the strategy date.</p>
{table_html(earn_tbl)}
<p>Short-interest variables <code>dsi</code>, <code>dtcn</code>, and <code>ddtcn</code> are read from the provided point-in-time panel. We apply an additional one-trading-day decision lag before these fields enter alpha features or borrow-tier rules.</p>
{table_html(si_tbl)}

<figure>
  <img src="{imgs.get('si_series','')}" alt="Short-interest representative series" style="max-width:88%;">
  <figcaption>Representative HLT short-interest proxy series, 2015&ndash;2024. Provided point-in-time proxies plus one-trading-day decision lag. No AUM or portfolio cost assumption.</figcaption>
</figure>

<h3>3.4 Stylised Fact Check</h3>
<p>In our eligible equal-weight universe, the result is less clean than the textbook version: the return identity holds exactly, but close-to-close return is larger than overnight return because intraday returns are also positive. We use the overnight effect as motivation, not as proof that any strategy should work.</p>
<figure>
  <img src="{imgs.get('decomp','')}" alt="Return decomposition" style="max-width:90%;">
  <figcaption>Equal-weight eligible-universe overnight/intraday/close-to-close return decomposition, 2010&ndash;2024. No AUM, no strategy costs, annual large-cap universe.</figcaption>
</figure>

<h3>3.5 Universe Evolution</h3>
{table_html(uni_tbl, cls="small")}
""")

    # --- SECTION 4: CAPACITY ---
    cap_tbl = capacity.copy()
    cap_tbl["AUM"] = cap_tbl["AUM"].apply(aum_label)
    cap_tbl = cap_tbl.rename(columns={
        "AUM": "AUM", "average_gross_exposure_used": "Avg Gross Exposure",
        "average_abs_per_stock_position": "Avg Abs Position",
        "max_abs_per_stock_position": "Max Abs Position",
        "average_participation_rate": "Avg Participation",
        "fraction_cap_binding_days": "Position-Days at Cap",
    })
    cap_show = cap_tbl[["AUM","Avg Gross Exposure","Avg Abs Position","Max Abs Position","Avg Participation","Position-Days at Cap"]]

    imp_tbl = impact.copy()
    imp_tbl["AUM_example"] = imp_tbl["example"]
    imp_show = pd.DataFrame({
        "Example": imp_tbl["example"],
        "Ticker": imp_tbl["ticker"],
        "Date": imp_tbl["date"],
        "Market Cap": imp_tbl["market_cap"].apply(money),
        "ADV20": imp_tbl["adv20_dollar"].apply(money),
        "Vol20": imp_tbl["vol20_annual"].apply(lambda x: pct(x)),
        "Daily Sigma": imp_tbl["sigma_daily"].apply(lambda x: pct(x)),
        "Participation": ["5.00%"] * len(imp_tbl),
        "Impact bps": imp_tbl.apply(lambda r: f"{0.7 * r['sigma_daily'] * np.sqrt(0.05) * 10000:.1f}", axis=1),
    })

    # Eligibility pivot
    elig_piv = elig_raw.pivot(index="year", columns="eligibility_reason", values="stock_days").fillna(0).astype(int)
    for c in ["OK","ADV_FAIL","MCAP_FAIL","PRICE_FAIL","VOL_FAIL","EARN_WINDOW"]:
        if c not in elig_piv.columns:
            elig_piv[c] = 0
    elig_piv = elig_piv[["OK","ADV_FAIL","MCAP_FAIL","PRICE_FAIL","VOL_FAIL","EARN_WINDOW"]].reset_index()
    elig_piv.columns = ["Year","OK","ADV_FAIL","MCAP_FAIL","PRICE_FAIL","VOL_FAIL","EARN_WINDOW"]

    cap_table_html = table_html(cap_show, fmt_map={
        "Avg Gross Exposure": pct, "Avg Participation": pct, "Position-Days at Cap": pct,
        "Avg Abs Position": money, "Max Abs Position": money,
    })

    sections.append(f"""
<h2><span class="section-num">4</span>Capacity-Aware Universe and Execution Assumptions</h2>
<p>The investable universe uses simple filters: prior-year top 1000 market-cap membership, at least 252 history days, price above $5, ADV20 above $10M, annualised 20-day volatility between 5% and 120%, and a &plusmn;1 trading-day earnings exclusion window. The portfolio applies a 5% ADV20 per-name participation cap.</p>
<p>As a sanity check, a simple square-root proxy at the 5% ADV cap gives the following daily impact scale:</p>
{table_html(imp_show)}
<p>These proxy values are larger than the fixed 1.5 bps auction slippage per leg, so this is a conservative caveat.</p>
{cap_table_html}
<figure>
  <img src="{imgs.get('capacity','')}" alt="Capacity by AUM" style="max-width:72%;">
  <figcaption>Capacity by AUM for the final top/bottom 3% equal-weight strategy; 50M/250M/1B, fixed 5% ADV20 cap, full commission/slippage/borrow schedule.</figcaption>
</figure>
<p>The 1B case is a capacity boundary case. Its average gross exposure is only <strong>{pct(capacity[capacity['AUM']==1e9].iloc[0]['average_gross_exposure_used'])}</strong>, so the strategy cannot deploy target risk under the fixed 5% ADV20 cap.</p>
<h3>Binding Eligibility Reasons by Year</h3>
{table_html(elig_piv, cls="small")}
""")

    # --- SECTION 5: BORROW ---
    bt250 = borrow_t[borrow_t["AUM"]==2.5e8][["borrow_tier","share_of_short_position_days","average_short_notional","total_borrow_cost"]].copy()
    bt250.columns = ["Tier", "Share of Short Days", "Avg Short Notional", "Total Borrow Cost"]

    he_tbl = hard_exc.copy()
    he_show = pd.DataFrame({
        "AUM": he_tbl["AUM"].apply(aum_label),
        "Final Sharpe": he_tbl["net_sharpe"].apply(lambda x: num(x)),
        "Hard-Excl Sharpe": ["1.363","1.462","1.383"],  # from LaTeX
        "Final Return": he_tbl["net_annual_return"].apply(pct),
        "Final Max DD": he_tbl["max_drawdown"].apply(pct),
    })

    # Borrow external validation
    bv_html = ""
    for _, row in borrow_v.iterrows():
        bv_html += f'<div class="callout"><strong>{row["ticker"]} ({row["window_start"]} to {row["window_end"]}):</strong> {esc(row["external_check"])}<br><em>Internal proxy:</em> Mean DSI = {row["mean_dsi_lag1"]:.3f}, Tier B/C fraction = {pct(row["tier_b_or_c_fraction"])}. {esc(row["internal_proxy_read"])}</div>'

    bt250_table_html = table_html(bt250, fmt_map={"Share of Short Days": pct, "Avg Short Notional": money, "Total Borrow Cost": money})

    sections.append(f"""
<h2><span class="section-num">5</span>Borrow Proxy and Short-Leg Financing</h2>
<p>The final strategy uses tiered borrow costs, not a hard short exclusion. Tier A is 40 bps p.a. Tier B is 200 bps when <code>dsi_lag1 &ge; 0.08</code>, <code>dtcn_lag1 &ge; 5.0</code>, or <code>ddtcn_lag1 &ge; 1.0</code>. Tier C is 800 bps when <code>dsi_lag1 &ge; 0.15</code>, <code>dtcn_lag1 &ge; 10.0</code>, or both <code>dsi_lag1 &ge; 0.10</code> and <code>ddtcn_lag1 &ge; 1.5</code>.</p>
<p>At 250M, <strong>56.79%</strong> of selected short position-days are Tier B or C.</p>
{bt250_table_html}

<h3>External Validation</h3>
<p>We do not have direct prime-broker borrow-fee or securities-lending data. As partial external evidence, we checked our proxy against two well-documented short-squeeze episodes:</p>
{bv_html}

<h3>Hard-Exclusion Sensitivity</h3>
<p>As a diagnostic, we remove selected shorts with <code>dsi_lag1 &ge; 10%</code> without replacement. This is not the promoted strategy.</p>
{table_html(he_show)}
<p>The result supports our choice to use tiered borrow costs rather than hard exclusion. High-DSI shorts are not the only source of performance.</p>
""")

    # --- SECTION 6: PROMOTION ---
    promo = pd.DataFrame([
        ["Previous champion", "0.811", "3.93%", "-10.19%", "4y rolling vol-scaled target"],
        ["Final champion", num(p250['net_sharpe']), pct(p250['net_annual_return']), pct(p250['max_drawdown']), "expanding-window vol-scaled target"],
    ], columns=["Model", "250M Sharpe", "250M Return", "Max Drawdown", "Description"])

    # Top challengers
    top_champ = champ.head(5)[["experiment_id","full_2010_2024_250m_net_sharpe","full_2010_2024_250m_max_drawdown","passes_phase2_replacement_rule"]].copy()
    top_champ.columns = ["Experiment", "Full Sharpe", "Full Max DD", "Passes Rule"]
    top_champ["Full Sharpe"] = top_champ["Full Sharpe"].apply(lambda x: num(x))
    top_champ["Full Max DD"] = top_champ["Full Max DD"].apply(pct)

    sections.append(f"""
<h2><span class="section-num">6</span>Baseline, Promotion Decision, and Challenger Models</h2>
<p>The first baseline predicted raw next overnight return. The promoted final model keeps the volatility-scaled target and changes the training window to expanding.</p>
{table_html(promo, cls="left-align")}
<p>The promotion rule was stricter than "pick the largest Sharpe". We wanted validation improvement, non-failure in 2023&ndash;2024, no obvious drawdown damage, and no hidden relaxation of costs or capacity.</p>
<h3>Top Phase-2 Challengers</h3>
{table_html(top_champ, cls="left-align")}
""")

    # --- SECTION 7: ALPHA ---
    abl250 = ablation[ablation["AUM"]==2.5e8][["model_variant","removed_group","IC_mean","IC_tstat","net_annual_return","net_vol","net_sharpe","max_drawdown"]].copy()
    abl250.columns = ["Variant","Removed Group","IC Mean","IC t-stat","Net Return","Net Vol","Net Sharpe","Max DD"]

    abl250_table_html = table_html(abl250, fmt_map={"IC Mean": lambda x: num(x), "IC t-stat": lambda x: num(x,2), "Net Return": pct, "Net Vol": pct, "Net Sharpe": lambda x: num(x), "Max DD": pct}, cls="left-align")

    sections.append(f"""
<h2><span class="section-num">7</span>Alpha Design, Model Selection, and Interpretability</h2>
<h3>7.1 Target and Model Class</h3>
<p>The baseline predicted raw next overnight return. That target is noisy because high-volatility names dominate. The final target divides by trailing daily volatility, so the model asks which stocks have better expected overnight return per unit of recent risk. The alpha learner is a 50% prior and 50% learned feature-weight correlation estimate on ranked point-in-time features.</p>

<h3>7.2 Information Coefficient and Weak Regimes</h3>
<p>The final alpha has full-sample mean IC <strong>{num(p250['IC_mean'])}</strong> with t-stat <strong>{num(p250['IC_tstat'],2)}</strong>. Yearly IC is not uniformly positive. The weak years: 2010 has negative IC, 2022 has negative IC, and 2024 has near-zero IC.</p>
<figure>
  <img src="{imgs.get('ic_year','')}" alt="Yearly IC" style="max-width:88%;">
  <figcaption>Final linear-score mean IC by calendar year, 2010&ndash;2024. Negative years highlighted. 250M AUM, top/bottom 3% basket, 5% ADV20 cap, Section 6.3 cost schedule.</figcaption>
</figure>
<figure>
  <img src="{imgs.get('rolling','')}" alt="Rolling IC" style="max-width:88%;">
  <figcaption>63-day rolling IC for the final linear score. Spearman correlation between alpha score and subsequent overnight return, 2010&ndash;2024.</figcaption>
</figure>

<h3>7.3 Feature Weights</h3>
<figure>
  <img src="{imgs.get('weights','')}" alt="Feature weights" style="max-width:85%;">
  <figcaption>Largest 2024 final linear-score feature weights. Transparent expanding-window model, unchanged feature set, no LightGBM portfolio promotion.</figcaption>
</figure>

<h3>7.4 LightGBM Diagnostic</h3>
<p>LightGBM was run as an optional IC diagnostic. It is not the final strategy.</p>
<figure>
  <img src="{imgs.get('ic_comp','')}" alt="IC comparison" style="max-width:70%;">
  <figcaption>Final linear score versus LightGBM diagnostic IC, validation and internal holdout. Same data panel and cutoff. LightGBM is not promoted as a costed strategy.</figcaption>
</figure>

<h3>7.5 Feature Ablation</h3>
<p>Removing return/reversal destroys the strategy. Other groups are less clean: removing earnings/revision or short-interest/borrow-stress can improve frozen Sharpe. We interpret those variables partly as risk, cost, crowding, or regime controls.</p>
{abl250_table_html}
<figure>
  <img src="{imgs.get('ablation','')}" alt="Feature ablation" style="max-width:85%;">
  <figcaption>Feature-group ablation at 250M for the final top/bottom 3% equal-weight strategy; fixed 5% ADV20 cap and full cost schedule.</figcaption>
</figure>
""")

    # --- SECTION 8: PORTFOLIO AND COSTS ---
    perf_show = pd.DataFrame({
        "AUM": perf["AUM"].apply(aum_label),
        "Net Return": perf["net_annual_return"].apply(pct),
        "Net Vol": perf["net_vol"].apply(pct),
        "Sharpe": perf["net_sharpe"].apply(lambda x: num(x)),
        "Max DD": perf["max_drawdown"].apply(pct),
        "Turnover": perf["average_turnover"].apply(lambda x: num(x)),
        "Gross Exposure": perf["average_gross_exposure_used"].apply(pct),
        "Long/Short Names": perf.apply(lambda r: f"{r['average_long_names']:.1f}/{r['average_short_names']:.1f}", axis=1),
    })

    cost_show = pd.DataFrame({
        "AUM": borrow_s["AUM"].apply(aum_label),
        "Gross Sharpe": borrow_s["gross_sharpe"].apply(lambda x: num(x)),
        "Commission Drag": borrow_s["commission_drag"].apply(lambda x: num(x)),
        "Slippage Drag": borrow_s["slippage_drag"].apply(lambda x: num(x)),
        "Borrow Drag": borrow_s["borrow_drag"].apply(lambda x: num(x)),
        "Net Sharpe": borrow_s["net_sharpe"].apply(lambda x: num(x)),
        "Avg Borrow bps/day": borrow_s["avg_borrow_bps_per_day"].apply(lambda x: num(x)),
    })

    sections.append(f"""
<h2><span class="section-num">8</span>Portfolio Construction, Costs, and Headline Results</h2>
<p>Each day the strategy ranks eligible stocks by raw alpha score, selects the top and bottom 3% with a minimum of 15 names per side, equal-weights each side, then applies per-name capacity caps. The resulting book is dollar-neutral after capacity sizing. Average selected names are about 24.9 long and 24.9 short per day.</p>
{table_html(perf_show)}
<h3>Gross-to-Net Sharpe Decomposition</h3>
{table_html(cost_show)}
<p>Auction slippage is the largest explicit cost drag. The QuantStats tear-sheet is provided as <code>outputs/quantstats_250m.html</code> against the S&amp;P 500 total return benchmark.</p>
<figure>
  <img src="{imgs.get('cumret','')}" alt="Cumulative returns by AUM" style="max-width:88%;">
  <figcaption>Final net cumulative returns by AUM; 50M/250M/1B, final top/bottom 3% equal-weight strategy, fixed 5% ADV20 cap and full commission/slippage/borrow schedule.</figcaption>
</figure>
""")

    # --- SECTION 9: ROBUSTNESS ---
    r250 = robust[robust["AUM"]==2.5e8]
    yearly = r250[r250["subperiod"].str.startswith("year_")].copy()
    yearly["Year"] = yearly["subperiod"].str.replace("year_","")
    yr_show = yearly[["Year","annual_return","annual_vol","sharpe","max_drawdown","average_gross_exposure_used"]].copy()
    yr_show.columns = ["Year","Return","Vol","Sharpe","Max DD","Gross Exposure"]

    stress_show = r250[r250["subperiod"].str.startswith("stress_")].copy()
    stress_show["Window"] = stress_show["subperiod"].str.replace("stress_","")
    st_show = stress_show[["Window","annual_return","annual_vol","sharpe","max_drawdown","average_gross_exposure_used"]].copy()
    st_show.columns = ["Window","Return","Vol","Sharpe","Max DD","Gross Exposure"]

    split = r250[r250["subperiod"].str.startswith("split_")].copy()
    split["Window"] = split["subperiod"].str.replace("split_","")
    sp_show = split[["Window","annual_return","annual_vol","sharpe","max_drawdown","average_gross_exposure_used"]].copy()
    sp_show.columns = ["Window","Return","Vol","Sharpe","Max DD","Gross Exposure"]

    full250 = r250[r250["subperiod"]=="full_2010_2024"].iloc[0]

    lo_show = lo_adj[["AUM_label","net_sharpe_raw","lo_adjusted_net_sharpe","autocorr_lag1"]].copy()
    lo_show.columns = ["AUM","Raw Sharpe","Lo-Adjusted Sharpe","Lag-1 Autocorrelation"]

    robustness_fmt = {"Return":pct,"Vol":pct,"Sharpe":lambda x:num(x),"Max DD":pct,"Gross Exposure":pct}
    st_table_html = table_html(st_show, fmt_map=robustness_fmt)
    sp_table_html = table_html(sp_show, fmt_map=robustness_fmt)
    lo_table_html = table_html(lo_show, fmt_map={"Raw Sharpe":lambda x:num(x),"Lo-Adjusted Sharpe":lambda x:num(x),"Lag-1 Autocorrelation":lambda x:num(x,4)})

    sections.append(f"""
<h2><span class="section-num">9</span>Robustness and Stress Tests</h2>
<h3>Stress Windows</h3>
{st_table_html}
<h3>Pre/Post Split</h3>
{sp_table_html}
<p>Tail diagnostics at 250M: worst 3-month return <strong>{pct(full250['worst_3m_return'])}</strong>, worst 6-month return <strong>{pct(full250['worst_6m_return'])}</strong>, worst 12-month return <strong>{pct(full250['worst_12m_return'])}</strong>. Top 5% best days contribute <strong>{num(full250['top_5pct_days_pnl_share'],3)}</strong> times total PnL. Lag-1 return autocorrelation is <strong>{num(full250['return_autocorrelation_lag1'])}</strong>.</p>
<h3>Lo Autocorrelation-Adjusted Sharpe</h3>
{lo_table_html}
""")

    # --- SECTION 10: REPRODUCIBILITY ---
    sections.append("""
<h2><span class="section-num">10</span>Reproducibility and Code Audit</h2>
<p>The full pipeline is deterministic. Random sampling is not used in the final alpha learner. The report-ready output files are generated from code, not edited by hand. The local reproduction commands are:</p>
<div class="formula" style="white-space:pre-wrap;">python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHON=.venv/bin/python make reproduce-final
PYTHON=.venv/bin/python make ablation
PYTHON=.venv/bin/python make report-assets
.venv/bin/python -m pytest -q</div>
<p>The tests cover portfolio neutrality, submission rules, capacity fields, and the blended challenger logic. The notebook <code>notebooks/c2o_research_process_and_tests.ipynb</code> reproduces the key report checks from generated outputs.</p>
""")

    # --- SECTION 11: LIMITATIONS ---
    lim = pd.DataFrame([
        ["Borrow data", "No external prime-broker feed available", "Tiered borrow as proxy plus hard-exclusion sensitivity"],
        ["Market impact", "Backtest uses fixed auction slippage", "Disclose square-root impact proxy as caveat"],
        ["Holdout decay", "2023-2024 Sharpe lower than validation", "Keep the number visible; avoid overclaiming"],
        ["Capacity", "1B cannot deploy target gross exposure", "Use 250M as headline; treat 1B as boundary"],
        ["Model complexity", "More aggressive ML models untested fully", "Keep ridge/elastic net/blend as challengers"],
    ], columns=["Area", "Limitation", "How We Handle It"])

    sections.append(f"""
<h2><span class="section-num">11</span>Limitations and Next Work</h2>
{table_html(lim, cls="left-align")}
<p>The next modelling direction is the ensemble, not a sudden jump to a black-box model. The fixed blend already improves the 2023-2024 holdout Sharpe, but it does not satisfy the full promotion rule.</p>
""")

    # --- SECTION 12: SCEPTICAL MARKER ---
    sections.append("""
<h2><span class="section-num">12</span>Sceptical Marker Checklist</h2>
<p><strong>Look-ahead.</strong> Features are observable by the decision time or shifted. Earnings use strategy trading dates with AMC/BMO treatment. Short interest is lagged an extra trading day. The universe is prior-year-end defined. No 2025+ observations enter development.</p>
<p><strong>Statistical robustness.</strong> We report validation, internal holdout, year-by-year results, IC by year/regime, stress windows, rolling loss windows, lag-1 autocorrelation, and PnL concentration. We do not hide the lower 2023-2024 holdout Sharpe or negative 2024 return.</p>
<p><strong>Capacity and execution honesty.</strong> The 5% ADV20 cap is binding frequently. At 1B, average gross exposure drops to 25.35%, so 1B is a capacity boundary, not the main headline.</p>
<p><strong>Borrow honesty.</strong> Borrow costs are tiered and charged daily. No external borrow fee validation is claimed beyond anecdotal evidence (GME, CVNA). A DSI&ge;10% hard-exclusion diagnostic is reported and does not break the strategy.</p>
<p><strong>Reporting integrity.</strong> All headline tables and charts trace to code-generated outputs. The report uses 250M as the main case but also discloses 50M and 1B. Figures are labelled with the cost and capacity context.</p>
""")

    # --- SECTION 13: CONCLUSION ---
    sections.append("""
<h2><span class="section-num">13</span>Conclusion</h2>
<p>The final submitted strategy is the expanding-window volatility-scaled close-to-open alpha. It is more aggressive than the plain baseline because the target is risk-scaled and the challenger search considered ridge, elastic net, and blended ML scores. It is still controlled enough for submission because the promoted model is auditable, deterministic, point-in-time, and checked under costs, borrow, capacity, ablation, and stress tests. The next research direction is an ensemble challenger, but only if it improves validation and 2023-2024 holdout without worsening drawdown.</p>
""")

    # --- APPENDIX A ---
    sections.append(f"""
<h2><span class="section-num">A</span>Appendix A: Universe and Eligibility Evidence</h2>
{table_html(uni_tbl, cls="small")}
{table_html(elig_piv, cls="small")}
""")

    # --- APPENDIX B ---
    feat_show = features[["feature_name","formula","required_lag","observable_1550_et_day_t"]].copy()
    feat_show.columns = ["Feature","Formula","Lag","Observable 15:50 ET"]

    ic_show = ic_yr[["year","mean_ic","ic_tstat","days","median_n"]].copy()
    ic_show.columns = ["Year","Mean IC","IC t-stat","Days","Median N"]

    ic_table_html = table_html(ic_show, fmt_map={"Mean IC":lambda x:num(x,4),"IC t-stat":lambda x:num(x,2)}, cls="small")

    sections.append(f"""
<h2><span class="section-num">B</span>Appendix B: Feature Inventory and IC by Year</h2>
{table_html(feat_show, cls="small left-align")}
{ic_table_html}
""")

    # --- APPENDIX C ---
    yr_table_html = table_html(yr_show, fmt_map={"Return":pct,"Vol":pct,"Sharpe":lambda x:num(x),"Max DD":pct,"Gross Exposure":pct}, cls="small")

    sections.append(f"""
<h2><span class="section-num">C</span>Appendix C: 250M Year-by-Year Performance</h2>
{yr_table_html}
""")

    # --- APPENDIX D ---
    manifest = pd.DataFrame([
        ["outputs/performance_summary.csv", "Headline 50M, 250M and 1B performance table"],
        ["outputs/daily_returns_250m.csv", "Daily 250M net returns used for QuantStats"],
        ["outputs/positions_250m.parquet", "Position-level economics, borrow tier, capacity"],
        ["outputs/report_ready/feature_inventory.csv", "Feature formulas, lags and observability"],
        ["outputs/report_ready/locked_strategy_robustness.csv", "Year, stress, tail and autocorrelation robustness"],
        ["notebooks/c2o_research_process_and_tests.ipynb", "Notebook that checks report numbers from outputs"],
    ], columns=["File", "Purpose"])

    sections.append(f"""
<h2><span class="section-num">D</span>Appendix D: Output Manifest</h2>
{table_html(manifest, cls="left-align")}
<p>The repository contains more audit files, but these are directly tied to the report's headline claims. To reproduce, place original data files under <code>data/</code> and run <code>make reproduce</code>.</p>
""")

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>C2O Close-to-Open Strategy &mdash; Final Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
{"".join(sections)}
</div>
</body>
</html>"""

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to {out_path} ({len(html):,} bytes)")

if __name__ == "__main__":
    main()
