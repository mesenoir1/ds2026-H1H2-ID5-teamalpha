from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="XAI-Guided Knee OA Grading Prototype",
    layout="wide",
)

try:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from prototype_utils import (
        BORDER_THRESHOLD,
        PROJECT_ROOT,
        ROI_THRESHOLD,
        CheckpointInfo,
        PredictionResult,
        choose_device,
        discover_checkpoints,
        explain_model,
        load_checkpoint_model,
        make_masks,
        mask_overlay,
        prediction_table,
        predict_probabilities,
        seed_prediction_row,
        predict_single,
        preprocess_image,
        summarize_probabilities,
    )
except Exception as exc:
    st.title("XAI-Guided Knee OA Grading Prototype")
    st.error(
        "The prototype could not import the project dependencies or the existing code/ modules. "
        "Use a clean virtual environment, install the repository requirements, and run the app from the project root."
    )
    st.code(
        ".\\.venv\\Scripts\\python.exe -m pip install --no-cache-dir -r requirements.txt\n"
        ".\\.venv\\Scripts\\python.exe -m streamlit run app.py\n\n"
        f"Original error: {exc}"
    )
    st.stop()


@st.cache_data(show_spinner=False)
def cached_discover_checkpoints(root: str) -> dict[int, CheckpointInfo]:
    return discover_checkpoints(root)


@st.cache_resource(show_spinner=False)
def cached_load_model(checkpoint_path: str, model_family: str, device_name: str):
    device = torch.device(device_name)
    return load_checkpoint_model(checkpoint_path, model_family=model_family, device=device)


def checkpoint_summary(checkpoints: dict[int, CheckpointInfo]) -> pd.DataFrame:
    rows = [
        {
            "seed": seed,
            "folder": str(info.folder.relative_to(PROJECT_ROOT)) if info.folder.is_relative_to(PROJECT_ROOT) else str(info.folder),
            "checkpoint": str(info.path.relative_to(PROJECT_ROOT)) if info.path.is_relative_to(PROJECT_ROOT) else str(info.path),
        }
        for seed, info in checkpoints.items()
    ]
    return pd.DataFrame(rows)


def predict_ensemble_cached(
    checkpoints: dict[int, CheckpointInfo],
    model_family: str,
    image_tensor: torch.Tensor,
    device_name: str,
) -> PredictionResult:
    rows = []
    probabilities = []
    for seed, checkpoint in sorted(checkpoints.items()):
        model = cached_load_model(str(checkpoint.path), model_family, device_name)
        probs = predict_probabilities(model, image_tensor)
        rows.append(seed_prediction_row(seed, probs))
        probabilities.append(probs)

    if not probabilities:
        raise ValueError(f"No checkpoints available for {model_family} ensemble prediction.")

    mean_probs = np.mean(np.stack(probabilities, axis=0), axis=0)
    pred_class, confidence = summarize_probabilities(mean_probs)
    return PredictionResult(
        probabilities=mean_probs,
        pred_class=pred_class,
        confidence=confidence,
        seed_rows=rows,
    )


def prediction_card(title: str, prediction, explanation) -> None:
    status = "Suspicious" if explanation.suspicious else "Not suspicious"
    status_method = st.error if explanation.suspicious else st.success

    st.subheader(title)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted KL grade", prediction.pred_class)
    col_b.metric("Confidence", f"{prediction.confidence:.1%}")
    col_c.metric("ROIInside", f"{explanation.roi_inside:.3f}")
    st.metric("BorderAttention", f"{explanation.border_attention:.3f}")
    status_method(f"{status}: {explanation.suspicion_reason}")


def format_seed_option(seed: int) -> str:
    return f"Seed {seed}"


st.title("XAI-Guided Knee OA Grading Prototype")
st.caption(
    "Research prototype for comparing DenseNet201 weighted cross-entropy baseline models "
    "with final XAI-guided DenseNet201 models. This is not a medical diagnostic tool."
)
st.info(
    "ROIInside and BorderAttention are proxy-based saliency alignment metrics, "
    "not clinically validated faithfulness measures."
)

device = choose_device()
device_name = str(device)

with st.sidebar:
    st.header("Settings")
    baseline_root = st.text_input(
        "Baseline model folder",
        value=str(PROJECT_ROOT / "baseline_39_45"),
    )
    final_root = st.text_input(
        "Final XAI-guided model folder",
        value=str(PROJECT_ROOT / "final_models_39_45"),
    )

    prediction_mode = st.radio(
        "Prediction mode",
        options=["single seed", "ensemble"],
        horizontal=False,
    )

    st.write(f"Device: `{device_name}`")
    if device.type == "cuda":
        st.caption(torch.cuda.get_device_name(0))

baseline_checkpoints = cached_discover_checkpoints(baseline_root)
final_checkpoints = cached_discover_checkpoints(final_root)
common_seeds = sorted(set(baseline_checkpoints) & set(final_checkpoints))

with st.sidebar:
    if not baseline_checkpoints:
        st.error("No baseline best_model.pt checkpoints were found.")
    if not final_checkpoints:
        st.error("No final-model best_model.pt checkpoints were found.")
    if baseline_checkpoints and final_checkpoints and not common_seeds:
        st.error("Baseline and final folders have no matching seeds.")

    seed_options = common_seeds or sorted(set(baseline_checkpoints) | set(final_checkpoints))
    if seed_options:
        default_seed = seed_options.index(39) if 39 in seed_options else 0
        selected_seed = st.selectbox(
            "Seed for single-seed prediction and Grad-CAM",
            options=seed_options,
            index=default_seed,
            format_func=format_seed_option,
        )
    else:
        selected_seed = None
        st.info("No seed selector is available until checkpoints are found.")

    with st.expander("Discovered baseline checkpoints"):
        st.dataframe(checkpoint_summary(baseline_checkpoints), hide_index=True, use_container_width=True)
    with st.expander("Discovered final checkpoints"):
        st.dataframe(checkpoint_summary(final_checkpoints), hide_index=True, use_container_width=True)

uploaded_file = st.file_uploader(
    "Upload a knee X-ray image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded_file is None:
    st.write("Upload an image to run the comparison.")
    st.stop()

if not seed_options:
    st.stop()

try:
    image = Image.open(uploaded_file)
except Exception as exc:
    st.error(f"Could not open the uploaded image: {exc}")
    st.stop()

try:
    image_tensor, image_np = preprocess_image(image, device)
    roi_mask, border_mask = make_masks(image_np)
except Exception as exc:
    st.error(f"Preprocessing or dynamic ROI generation failed: {exc}")
    st.stop()

if prediction_mode == "single seed":
    if selected_seed not in baseline_checkpoints or selected_seed not in final_checkpoints:
        st.error(f"Seed {selected_seed} is not available for both model families.")
        st.stop()
    baseline_prediction_checkpoints = {selected_seed: baseline_checkpoints[selected_seed]}
    final_prediction_checkpoints = {selected_seed: final_checkpoints[selected_seed]}
else:
    baseline_prediction_checkpoints = baseline_checkpoints
    final_prediction_checkpoints = final_checkpoints
    st.caption(
        "Prediction is ensembled across available seeds. Grad-CAM and ROI metrics are displayed "
        "for the selected seed because explanations are seed-specific."
    )

if selected_seed not in baseline_checkpoints or selected_seed not in final_checkpoints:
    st.error(f"Selected Grad-CAM seed {selected_seed} is not available for both model families.")
    st.stop()

try:
    with st.spinner("Loading models and running predictions..."):
        if prediction_mode == "single seed":
            baseline_model = cached_load_model(
                str(baseline_checkpoints[selected_seed].path),
                "baseline",
                device_name,
            )
            final_model = cached_load_model(
                str(final_checkpoints[selected_seed].path),
                "final",
                device_name,
            )
            baseline_prediction = predict_single(baseline_model, image_tensor, selected_seed)
            final_prediction = predict_single(final_model, image_tensor, selected_seed)
        else:
            baseline_prediction = predict_ensemble_cached(
                baseline_prediction_checkpoints,
                model_family="baseline",
                image_tensor=image_tensor,
                device_name=device_name,
            )
            final_prediction = predict_ensemble_cached(
                final_prediction_checkpoints,
                model_family="final",
                image_tensor=image_tensor,
                device_name=device_name,
            )
            baseline_model = cached_load_model(
                str(baseline_checkpoints[selected_seed].path),
                "baseline",
                device_name,
            )
            final_model = cached_load_model(
                str(final_checkpoints[selected_seed].path),
                "final",
                device_name,
            )

    with st.spinner("Generating predicted-class Grad-CAM and saliency alignment metrics..."):
        baseline_explanation = explain_model(
            baseline_model,
            image_tensor,
            image_np,
            roi_mask,
            border_mask,
        )
        final_explanation = explain_model(
            final_model,
            image_tensor,
            image_np,
            roi_mask,
            border_mask,
        )
except Exception as exc:
    st.error(f"Model inference or explanation failed: {exc}")
    st.stop()

st.divider()
st.subheader("Baseline vs Final Model Comparison")
col_baseline, col_final = st.columns(2)
with col_baseline:
    prediction_card("Baseline: DenseNet201 weighted CE", baseline_prediction, baseline_explanation)
with col_final:
    prediction_card("Final: XAI-guided DenseNet201", final_prediction, final_explanation)

st.caption(
    f"Suspicious prediction rule: ROIInside < {ROI_THRESHOLD:.2f} OR "
    f"BorderAttention > {BORDER_THRESHOLD:.2f}."
)

if prediction_mode == "ensemble":
    st.subheader("Individual Seed Predictions")
    col_seed_a, col_seed_b = st.columns(2)
    with col_seed_a:
        st.write("Baseline seeds")
        st.dataframe(
            prediction_table(baseline_prediction.seed_rows),
            hide_index=True,
            use_container_width=True,
        )
    with col_seed_b:
        st.write("Final-model seeds")
        st.dataframe(
            prediction_table(final_prediction.seed_rows),
            hide_index=True,
            use_container_width=True,
        )

st.subheader("Visualizations")
roi_overlay = mask_overlay(image_np, roi_mask, color=(1.0, 0.05, 0.05))
border_overlay = mask_overlay(image_np, border_mask, color=(1.0, 0.65, 0.0))

viz_cols = st.columns(3)
viz_cols[0].image(image_np, caption="Original X-ray (preprocessed 224x224)", clamp=True)
viz_cols[1].image(roi_overlay, caption="Dynamic ROI mask", clamp=True)
viz_cols[2].image(border_overlay, caption="Border mask", clamp=True)

overlay_cols = st.columns(2)
overlay_cols[0].image(
    baseline_explanation.overlay,
    caption=f"Baseline predicted-class Grad-CAM, seed {selected_seed}",
    clamp=True,
)
overlay_cols[1].image(
    final_explanation.overlay,
    caption=f"Final predicted-class Grad-CAM, seed {selected_seed}",
    clamp=True,
)
