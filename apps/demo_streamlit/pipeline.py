"""
News Structurizer demo (Streamlit)

Runs the current package pipeline (`news_structurizer`) on input text using local models
from `models/`.

Run:
  venv/bin/streamlit run apps/demo_streamlit/pipeline.py
"""

# ruff: noqa: E402

import json
import sys
from pathlib import Path

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _default_models() -> dict[str, str]:
    return {
        "segmenter": str(REPO_ROOT / "models" / "segmenter"),
        "topic": str(REPO_ROOT / "models" / "topic"),
        "scale": str(REPO_ROOT / "models" / "scale"),
        "extractor": str(REPO_ROOT / "models" / "extractor"),
    }


def _path_exists(path: str) -> bool:
    return Path(path).exists()


def _get_device_options() -> tuple[list[str], str]:
    try:
        import torch

        if torch.cuda.is_available():
            return ["auto", "cpu", "cuda:0"], "cuda:0"
    except Exception:
        pass
    return ["auto", "cpu"], "cpu"


@st.cache_resource
def load_pipeline(
    *,
    segmenter_model: str,
    topic_model: str,
    scale_model: str,
    extractor_model: str,
    device: str,
):
    from news_structurizer import Pipeline, PipelineConfig

    cfg = PipelineConfig(
        segmenter_model_path=segmenter_model,
        topic_model_path=topic_model,
        scale_model_path=scale_model,
        extractor_model_path=extractor_model,
        device=None if device == "auto" else device,
    )
    return Pipeline(cfg)


def main() -> None:
    st.set_page_config(page_title="News Structurizer", layout="wide")

    st.title("News Structurizer (demo)")
    st.caption("Text → segmentation → topic/scale → attribute extraction")

    defaults = _default_models()
    device_options, default_device = _get_device_options()

    with st.sidebar:
        st.header("Models")
        segmenter_model = st.text_input("Segmenter model path", value=defaults["segmenter"])
        topic_model = st.text_input("Topic model path", value=defaults["topic"])
        scale_model = st.text_input("Scale model path", value=defaults["scale"])
        extractor_model = st.text_input("Extractor model path", value=defaults["extractor"])

        st.header("Runtime")
        device = st.selectbox("Device", device_options, index=device_options.index(default_device))

        missing = [
            name
            for name, path in {
                "segmenter": segmenter_model,
                "topic": topic_model,
                "scale": scale_model,
                "extractor": extractor_model,
            }.items()
            if not _path_exists(path)
        ]
        if missing:
            st.error(f"Missing model directories: {', '.join(missing)}")

    default_text = (
        "бүгін астанада ауа райы суық болады түнде температура минус он бес градусқа дейін түседі "
        "ал енді спорт жаңалықтарына көшейік алматыдағы футбол ойынында қайрат командасы актобені "
        "үш бір есебімен ұтты шешуші голды екінші таймда соқты"
    )
    text_input = st.text_area("Input text", value=default_text, height=180)

    col1, col2 = st.columns([1, 6])
    with col1:
        run_btn = st.button("Run", type="primary")
    with col2:
        st.write("")

    if run_btn:
        if not text_input.strip():
            st.warning("Empty input.")
            return
        if missing:
            st.error("Fix missing model paths first.")
            return

        with st.spinner("Loading pipeline..."):
            pipeline = load_pipeline(
                segmenter_model=segmenter_model,
                topic_model=topic_model,
                scale_model=scale_model,
                extractor_model=extractor_model,
                device=device,
            )

        with st.spinner("Processing..."):
            report = pipeline.process_text(text_input.strip())

        st.success(f"News items: {len(report.news)}")

        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        st.download_button(
            "Download report.json",
            data=payload,
            file_name="report.json",
            mime="application/json",
        )

        for item in report.news:
            with st.expander(f"#{item.id}: {item.title[:80]}", expanded=(item.id == 1)):
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Topic", item.topic, f"{item.topic_confidence:.1%}")
                with c2:
                    st.metric("Scale", item.scale, f"{item.scale_confidence:.1%}")

                st.divider()
                st.markdown(f"**Title:** {item.title}")
                st.markdown(f"**Key events:** {item.key_events}")
                st.markdown(f"**Location:** {item.location}")
                st.markdown(f"**Key names:** {item.key_names}")

                st.divider()
                st.markdown("**Text:**")
                st.write(item.text)

        with st.expander("Raw JSON", expanded=False):
            st.code(payload, language="json")


if __name__ == "__main__":
    main()
