import tempfile
import streamlit as st
from google import genai

from core.state import BidState
from agents.extractor import extractor_agent, _get_client, _wait_until_active


st.set_page_config(page_title="Extractor Only (Gemini)", layout="wide")
st.title("🧪 Extractor-Only Test (Gemini Files + Structured JSON)")

with st.sidebar:
    st.subheader("Settings")
    st.session_state["MODEL_NAME"] = st.text_input("Model", value="gemini-2.5-flash")
    st.caption("Tip: use gemini-2.5-flash for testing; switch to gemini-2.5-pro later.")

client = _get_client()

uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded:
    if st.button("📤 Upload to Gemini + Run Extractor", type="primary"):
        state = BidState()

        st.write("📤 Uploading files to Gemini...")
        gemini_files = []
        for uf in uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getbuffer())
                tmp_path = tmp.name

            f = client.files.upload(path=tmp_path)
            st.success(f"Uploaded: {uf.name}")
            gemini_files.append(f)

        # wait active
        active = []
        for f in gemini_files:
            active.append(_wait_until_active(client, f))
        state.gemini_files = active

        st.write("✅ Files are ACTIVE. Running extractor...")
        state = extractor_agent(state)

        st.subheader("Results")
        if state.errors:
            st.error("⚠️ Errors occurred")
            for e in state.errors:
                st.code(e)
        else:
            st.success("✅ Extractor succeeded")
            st.write("**fileClassification:**", state.file_classification)
            st.write("**projectInfo:**", state.project_info)
            st.write("**projectReport:**")
            st.write(state.project_report)

            st.expander("Raw Extractor Output").write(state.raw_extractor_output)

            st.write("**Drawings files count:**", len(state.drawings_files))
            st.write("**Other files count:**", len(state.other_files))
