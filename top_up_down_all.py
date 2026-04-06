import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Model206 ALL", page_icon="🧬", layout="wide")
st.title("🧬 miRNA Upregulation Predictor — Model 206 ALL")
st.caption("LightGBM · Target Encoding · All conservation levels · family_conservation feature")

# ── Organism → miRNA prefix ──────────────────────────────────────────────────
ORGANISM_PREFIX = {"Human": "hsa-", "Mouse": "mmu-"}

# ── Load model bundle ────────────────────────────────────────────────────────
MODEL_FILE = "Model206_ALL_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_FILE, 'rb') as f:
        return pickle.load(f)

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file `{MODEL_FILE}` not found. Place it in the same directory as this script.")
    st.stop()

bundle           = load_model()
model            = bundle['model']
mirna_lookup     = bundle['mirna_lookup']
accession_lookup = bundle['accession_lookup']
options          = bundle['options']
metrics          = bundle['metrics']

# ── Metrics banner ───────────────────────────────────────────────────────────
st.markdown("#### Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}")
c2.metric("Accuracy", f"{metrics['acc_mean']:.3f}")
c3.metric("F1",       f"{metrics['f1_mean']:.3f}")
st.divider()

# ── Helper ───────────────────────────────────────────────────────────────────
def build_input_row_all(family, conservation, parasite, organism, cell_type, time_val):
    fam_val  = None if (not family or family == 'not_found') else family
    cons_val = float(conservation) if conservation is not None else None

    return pd.DataFrame([{
        'parasite':            parasite,
        'organism':            organism,
        'cell type':           cell_type,
        'family_name':         fam_val,
        'parasite_celltype':   f"{parasite.strip()}_{cell_type.strip()}",
        'time':                int(time_val),
        'is_found':            0 if fam_val is None else 1,
        'family_conservation': cons_val,
    }])

# ════════════════════════════════════════════════════════════════════════════
# MAIN — Top miRNA Rankings
# ════════════════════════════════════════════════════════════════════════════

st.subheader("📊 Top miRNAs Predicted Up or Down Under Your Conditions")
st.caption(
    "The model will score ALL miRNAs matching the selected organism using "
    "conservation-aware features (family + conservation level) and return "
    "the most confident predictions."
)

# ── Inputs ───────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

rl_parasite = col_a.selectbox(
    "Parasite",
    options.get('parasite', ["L.major", "L.donovani", "L.amazonensis", "L. donovani"])
)
rl_organism = col_b.selectbox(
    "Organism",
    options.get('organism', ["Human", "Mouse"])
)
rl_cell = col_a.selectbox(
    "Cell type",
    options.get('cell_type', ["PBMC", "THP-1", "BMDM (BALB/c females)",
                               "RAW 264.7", "Blood serum + liver (BALB/c )"])
)
rl_time = col_b.number_input(
    "Time (hours post-infection)", min_value=1, value=24
)
rl_top_n = st.slider(
    "Number of miRNAs to show per direction", 5, 30, 10
)

# ── Ranking Button ────────────────────────────────────────────────────────────
if st.button("🔍 Rank All miRNAs", type="primary"):

    prefix = ORGANISM_PREFIX.get(rl_organism, "")

    # ── ORGANISM FILTER (fixes mmu/hsa mismatch) ─────────────────────────────
    filtered = {
        name: entry
        for name, entry in mirna_lookup.items()
        if not prefix or name.lower().startswith(prefix.lower())
    }

    if not filtered:
        st.warning(f"No `{prefix}` miRNAs found in the database for organism **{rl_organism}**.")
        st.stop()

    rows = []
    for mirna_name, entry in filtered.items():
        family       = entry.get('family_name', None)
        conservation = entry.get('family_conservation', None)

        input_df = build_input_row_all(
            family, conservation,
            rl_parasite, rl_organism, rl_cell, rl_time
        )

        proba     = model.predict_proba(input_df)[0][1]
        direction = "⬆️ Up" if proba >= 0.5 else "⬇️ Down"

        fam_display = family if (family and family != 'not_found') else "Unknown"

        rows.append({
            "miRNA":          mirna_name,
            "Family":         fam_display,
            "Conservation":   conservation,
            "Parasite":       rl_parasite,
            "Cell Type":      rl_cell,
            "Time (h)":       int(rl_time),
            "Direction":      direction,
            "P(up)":          proba,
            "Confidence (%)": round(max(proba, 1 - proba) * 100, 1),
        })

    df_all = pd.DataFrame(rows)

    # ── Ranking ───────────────────────────────────────────────────────────────
    top_up = (
        df_all[df_all["P(up)"] >= 0.5]
        .sort_values("P(up)", ascending=False)
        .head(rl_top_n)
        .reset_index(drop=True)
    )
    top_up.index += 1

    top_down = (
        df_all[df_all["P(up)"] < 0.5]
        .sort_values("P(up)", ascending=True)
        .head(rl_top_n)
        .reset_index(drop=True)
    )
    top_down.index += 1

    DISPLAY_COLS = ["miRNA", "Family", "Conservation", "Parasite", "Cell Type", "Time (h)", "Confidence (%)"]

    # ── Display ───────────────────────────────────────────────────────────────
    st.markdown(f"**Conditions:** {rl_parasite} · {rl_organism} · "
                f"{rl_cell} · {rl_time}h post-infection")
    st.markdown(f"Scored **{len(df_all)} `{prefix}` miRNAs** — showing top {rl_top_n} per direction.")
    st.divider()

    col_up, col_down = st.columns(2)

    with col_up:
        st.markdown("### ⬆️ Top Upregulated")
        if top_up.empty:
            st.info("No miRNAs predicted as upregulated.")
        else:
            st.dataframe(top_up[DISPLAY_COLS], use_container_width=True)

    with col_down:
        st.markdown("### ⬇️ Top Downregulated")
        if top_down.empty:
            st.info("No miRNAs predicted as downregulated.")
        else:
            st.dataframe(top_down[DISPLAY_COLS], use_container_width=True)

    st.divider()
