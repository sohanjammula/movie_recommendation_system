# app.py – Streamlit UI (robust version)

import streamlit as st
import sys
import traceback

# --------------------------------------------------------------
# Try to import the recommender logic.
# If the artefact has an unexpected format, the loader will raise
# a RuntimeError – we catch it and show a helpful message.
# --------------------------------------------------------------
try:
    from src.recommender.model import recommend, df as catalogue_df
    _model_loaded = True
    _bundle_type = getattr(sys.modules["src.recommender.model"], "_bundle_type", "unknown")
except Exception as exc:        # any import / unpickle problem ends up here
    _model_loaded = False
    _load_error = exc
    _trace = traceback.format_exc()

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Content‑Based Movie Recommender")
st.caption(
    "Enter an exact movie title and get similar titles. "
    "If you see an error below, it means the model artefact "
    "`artifacts/recommender.pkl` is not in the expected format."
)

# ----------------------------------------------------------------------
# Show a clear error panel when the model cannot be loaded
# ----------------------------------------------------------------------
if not _model_loaded:
    st.error(
        "❗ Unable to load the recommender model.\n\n"
        f"**Error type:** `{type(_load_error).__name__}`\n"
        f"**Message:** `{_load_error}`\n\n"
        "**Typical causes**:\n"
        "1️⃣  The artefact was created with an old notebook that only saved\n"
        "    `new_df.to_dict()` and the similarity matrix.\n"
        "2️⃣  The artefact was pickled with a different pandas / scikit‑learn version.\n\n"
        "**What you can do**:\n"
        "- **If you still have the original CSV files**, run the new training script\n"
        "  `python src/train.py`.  It will produce a *new* `artifacts/recommender.pkl`\n"
        "  containing the dictionary expected by the UI.\n"
        "- **If you only have the old artefact**, the UI will still work (a little slower)\n"
        "  because it rebuilds a nearest‑neighbour index from the stored similarity matrix.\n"
        "- **If you prefer not to rebuild**, install the same pandas / scikit‑learn versions\n"
        "  that were used when the artefact was created (e.g. `pip install pandas==2.1.* scikit-learn==1.8.*`).\n\n"
        "Below is the original traceback to help you debug (optional):\n\n"
        f"```\n{_trace}\n```"
    )
    st.stop()   # stop the script – the rest of the UI cannot work without a model

# ----------------------------------------------------------------------
# Normal UI (model successfully loaded)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Search")
    title_input = st.text_input("Movie title (exact)", "")
    top_n = st.slider("How many results", 5, 30, 10)
    min_rating = st.slider("Minimum rating (0‑10)", 0.0, 10.0, 0.0, 0.1)
    min_year = st.slider("Earliest year", 1900, 2025, 2000)
    submit = st.button("Show recommendations")

if submit:
    if not title_input:
        st.error("Please type a movie title.")
    else:
        with st.spinner("🔎 Finding similar movies…"):
            try:
                recs = recommend(
                    title_input,
                    top_n=top_n,
                    min_rating=min_rating if min_rating > 0 else None,
                    min_year=min_year if min_year > 1900 else None,
                )
                if not recs:
                    st.info("No movies matched the selected filters.")
                else:
                    st.success(f"✅ Found **{len(recs)}** recommendations")
                    for i, rec in enumerate(recs, start=1):
                        st.markdown(
                            f"**{i}. {rec['title']}**  "
                            f"( {rec['year']} ) – "
                            f"⭐ {rec['vote_average']:.1f} – "
                            f"Score: {rec['hybrid_score']:.3f}"
                        )
            except Exception as e:
                # Any runtime error (e.g. unknown title) is caught here
                st.error(f"❌ Recommendation failed: {type(e).__name__}: {e}")

# ----------------------------------------------------------------------
# Small catalogue preview – useful for debugging / sanity check
# ----------------------------------------------------------------------
with st.expander("📋 First 10 titles in the catalogue (for reference)"):
    st.dataframe(
        catalogue_df[["title", "year", "vote_average"]].head(10)
    )


