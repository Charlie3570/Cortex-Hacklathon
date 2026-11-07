import os
import ast
import base64
import html
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.patches import Rectangle, Circle


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = "/content/AI_CT_Analyzer"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

CSV_PATH = os.path.join(
    RESULTS_DIR, "candidate_analysis_results.csv"
)

DESCRIPTION_PATH = os.path.join(
    RESULTS_DIR, "candidate_descriptions.csv"
)

VOLUME_PATH = os.path.join(
    RESULTS_DIR, "processed_ct_volume.npy"
)

DETECTION_PATH = os.path.join(
    RESULTS_DIR, "detection_data.npz"
)

ORTHOGONAL_PATH = os.path.join(
    RESULTS_DIR, "orthogonal_ct_views.png"
)

CONFIDENCE_PATH = os.path.join(
    RESULTS_DIR, "confidence_comparison.png"
)

HIGHEST_PATH = os.path.join(
    RESULTS_DIR, "highest_confidence_candidate.png"
)

ANALYTICS_PATH = os.path.join(
    RESULTS_DIR, "analytics_overview.png"
)

GALLERY_PATH = os.path.join(
    RESULTS_DIR, "candidate_gallery.png"
)

st.set_page_config(
    page_title="AI Lung CT Analyzer",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# 2. DARK DASHBOARD STYLING
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #080f1c;
    color: #e5eaf3;
}

header[data-testid="stHeader"] {
    background-color: #080f1c;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f1f5f9 !important;
}

p, li {
    color: #cbd5e1;
}

hr {
    border-color: #263449;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def image_to_base64(path):

    if not os.path.exists(path):
        return None

    with open(path, "rb") as file:
        return base64.b64encode(
            file.read()
        ).decode("utf-8")


def display_image_html(path, title):

    encoded = image_to_base64(path)

    if encoded is None:
        st.warning(f"Image not found: {title}")
        return

    extension = os.path.splitext(path)[1].lower()

    mime = "image/png"

    if extension == ".jpg" or extension == ".jpeg":
        mime = "image/jpeg"

    st.html(f"""
        <div style="
            background:#111c2e;
            border:1px solid #263449;
            border-radius:12px;
            padding:15px;
            margin:12px 0;
        ">

            <h3 style="
                color:#e2e8f0;
                font-size:17px;
                margin-top:0;
            ">
                {html.escape(title)}
            </h3>

            <img
                src="data:{mime};base64,{encoded}"
                style="
                    width:100%;
                    height:auto;
                    border-radius:8px;
                    display:block;
                "
            />

        </div>
    """)


@st.cache_resource
def load_ct_volume():

    return np.load(
        VOLUME_PATH,
        mmap_mode="r"
    )


@st.cache_resource
def load_detection_data():

    data = np.load(DETECTION_PATH)

    return {
        "boxes": data["boxes"],
        "scores": data["scores"],
        "labels": data["labels"],
        "affine": data["affine"]
    }


def parse_detection_ids(value):

    try:
        return ast.literal_eval(str(value))
    except Exception:
        return []


def create_candidate_image(
    volume,
    boxes,
    detection_ids,
    best_detection_index
):

    best_box = boxes[best_detection_index]

    # Determine the centre of the best detection.
    center_x = (
        best_box[0] + best_box[3]
    ) / 2

    center_y = (
        best_box[1] + best_box[4]
    ) / 2

    center_z = (
        best_box[2] + best_box[5]
    ) / 2

    slice_index = int(round(center_z))

    slice_index = max(
        0,
        min(slice_index, volume.shape[2] - 1)
    )

    # Extract the axial CT slice.
    ct_slice = volume[:, :, slice_index].T

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    fig.patch.set_facecolor("#080f1c")
    ax.set_facecolor("#080f1c")

    ax.imshow(
        ct_slice,
        cmap="gray",
        origin="lower"
    )

    visible_count = 0

    for detection_number in detection_ids:

        detection_index = int(detection_number) - 1

        if not 0 <= detection_index < len(boxes):
            continue

        box = boxes[detection_index]

        x1, y1, z1, x2, y2, z2 = box

        # Only draw boxes that intersect the selected slice.
        if not (z1 <= slice_index <= z2):
            continue

        width = x2 - x1
        height = y2 - y1

        rectangle = Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor="#ff4545",
            facecolor="none"
        )

        ax.add_patch(rectangle)

        detection_center_x = (x1 + x2) / 2
        detection_center_y = (y1 + y2) / 2

        circle = Circle(
            (detection_center_x, detection_center_y),
            radius=4,
            edgecolor="#ffdf45",
            facecolor="none",
            linewidth=2
        )

        ax.add_patch(circle)

        ax.annotate(
            f"Detection {detection_number}",
            (x1, y2),
            color="#ffdf45",
            fontsize=8,
            xytext=(3, 5),
            textcoords="offset points"
        )

        visible_count += 1

    ax.set_title(
        f"Candidate inspection — Axial slice {slice_index}",
        color="white",
        fontsize=14,
        pad=15
    )

    ax.set_xlabel(
        "X voxel coordinate",
        color="white"
    )

    ax.set_ylabel(
        "Y voxel coordinate",
        color="white"
    )

    ax.tick_params(colors="#94a3b8")

    for spine in ax.spines.values():
        spine.set_color("#475569")

    plt.tight_layout()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=130,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )

    plt.close(fig)

    buffer.seek(0)

    encoded = base64.b64encode(
        buffer.read()
    ).decode("utf-8")

    return encoded, slice_index, visible_count


# ============================================================
# 4. LOAD PROJECT DATA
# ============================================================

required_files = [
    CSV_PATH,
    VOLUME_PATH,
    DETECTION_PATH
]

missing_files = [
    path for path in required_files
    if not os.path.exists(path)
]

if missing_files:

    st.error("Some required project files are missing.")

    for path in missing_files:
        st.write(path)

    st.stop()


results = pd.read_csv(CSV_PATH)

if os.path.exists(DESCRIPTION_PATH):

    descriptions = pd.read_csv(DESCRIPTION_PATH)

    if "AI_Interpretation" in descriptions.columns:
        results["AI_Interpretation"] = (
            descriptions["AI_Interpretation"].values
        )

else:

    results["AI_Interpretation"] = (
        "The AI identified a possible candidate region. "
        "This is not a medical diagnosis."
    )


volume = load_ct_volume()

detection_data = load_detection_data()

boxes = detection_data["boxes"]

scores = detection_data["scores"]


# ============================================================
# 5. HEADER
# ============================================================

st.html("""
    <div style="margin-bottom:25px;">

        <div style="
            color:#60a5fa;
            font-size:13px;
            font-weight:700;
            letter-spacing:3px;
            margin-bottom:10px;
        ">
            COMPUTER-ASSISTED CT RESEARCH
        </div>

        <h1 style="
            font-size:38px;
            color:#f8fafc;
            margin:0;
        ">
            🫁 AI Lung CT Analyzer
        </h1>

        <p style="
            color:#94a3b8;
            font-size:16px;
            margin-top:10px;
        ">
            Explore possible lung nodule candidates identified
            by an experimental 3D AI detection model.
        </p>

        <div style="
            background:#422006;
            border:1px solid #92400e;
            color:#fde68a;
            padding:12px;
            border-radius:8px;
            margin-top:18px;
            font-size:13px;
        ">
            RESEARCH PROTOTYPE — AI detections are not confirmed
            nodules, medical diagnoses, or cancer probabilities.
        </div>

    </div>
""")


# ============================================================
# 6. ANALYSIS OVERVIEW
# ============================================================

st.divider()
st.header("CT Scan")
ct_source_path = os.path.join(PROJECT_DIR, "data", "scan_resampled.nii.gz")
if os.path.exists(ct_source_path):
    st.success("CT scan loaded and analyzed for this research demonstration.")
    st.caption(f"Source file: scan_resampled.nii.gz · {os.path.getsize(ct_source_path) / (1024**2):.2f} MB")
else:
    st.warning("Source CT file is not present in the project data folder.")


st.header("Analysis Overview")

raw_count = len(scores)

group_count = len(results)

highest_score = float(results["Confidence"].max())

average_score = float(results["Confidence"].mean())

cards = [
    ("Raw detections", raw_count, "#38bdf8"),
    ("Candidate groups", group_count, "#a78bfa"),
    ("Highest AI score", f"{highest_score:.4f}", "#34d399"),
    ("Average AI score", f"{average_score:.4f}", "#fbbf24")
]

card_html = ""

for label, value, accent in cards:

    card_html += f"""
        <div style="
            background:#111c2e;
            border:1px solid #263449;
            border-top:3px solid {accent};
            border-radius:12px;
            padding:18px;
            min-width:180px;
            flex:1;
        ">

            <div style="
                color:#94a3b8;
                font-size:13px;
                margin-bottom:10px;
            ">
                {html.escape(str(label))}
            </div>

            <div style="
                color:#f8fafc;
                font-size:27px;
                font-weight:700;
            ">
                {html.escape(str(value))}
            </div>

        </div>
    """

st.html(f"""
    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:12px;
        margin:15px 0 30px 0;
    ">
        {card_html}
    </div>
""")


# ============================================================
# 7. EXPLANATION
# ============================================================

st.header("What is the AI looking for?")

st.markdown("""
This system analyzes a 3D chest CT image and identifies regions
whose image patterns resemble those of lung nodules in its
training data.

A detection means the model has highlighted a region for
further inspection. It does not establish that the region is
a real nodule or indicate whether it is benign or malignant.
""")


# ============================================================
# 8. INTERACTIVE CANDIDATE EXPLORER
# ============================================================

st.divider()

st.header("Interactive Candidate Explorer")

st.write(
    "Choose a candidate to inspect its detection location "
    "and associated measurements."
)

candidate_numbers = [
    int(number)
    for number in results["Group"].tolist()
]

# Read candidate selection from the URL.
candidate_param = st.query_params.get(
    "candidate",
    str(candidate_numbers[0])
)

try:

    selected_group = int(candidate_param)

except (ValueError, TypeError):

    selected_group = candidate_numbers[0]

if selected_group not in candidate_numbers:

    selected_group = candidate_numbers[0]


# Create custom HTML candidate selection buttons.
# These replace Streamlit's selectbox component.

candidate_buttons = ""

for number in candidate_numbers:

    if number == selected_group:

        background = "#2563eb"
        border = "#60a5fa"

    else:

        background = "#1e293b"
        border = "#334155"

    candidate_buttons += f"""
        <a href="?candidate={number}"
           style="
               display:inline-block;
               padding:10px 15px;
               margin:5px;
               border-radius:8px;
               background:{background};
               border:1px solid {border};
               color:white;
               text-decoration:none;
               font-weight:600;
               font-size:13px;
           ">
           Candidate {number}
        </a>
    """


st.html(f"""
    <div style="
        background:#0f172a;
        padding:15px;
        border-radius:12px;
        margin:10px 0 20px 0;
        border:1px solid #263449;
    ">

        <div style="
            color:#cbd5e1;
            margin-bottom:8px;
            font-size:14px;
        ">
            Select a candidate to inspect:
        </div>

        {candidate_buttons}

    </div>
""")


# Get selected candidate row.
row = results[
    results["Group"] == selected_group
].iloc[0]


detection_ids = parse_detection_ids(
    row["Original_Detections"]
)


# Find the highest-scoring original detection
# belonging to the selected group.

valid_indices = [
    int(number) - 1
    for number in detection_ids
    if 0 <= int(number) - 1 < len(scores)
]

if not valid_indices:

    st.error("No valid detections were found for this candidate.")

    st.stop()


best_detection_index = max(
    valid_indices,
    key=lambda index: scores[index]
)


# Generate the selected candidate's CT visualization.

encoded_image, slice_index, visible_count = (
    create_candidate_image(
        volume,
        boxes,
        detection_ids,
        best_detection_index
    )
)


st.html(f"""
    <div style="
        color:#94a3b8;
        font-size:13px;
        margin-bottom:10px;
    ">
        Candidate {selected_group}
        · Axial slice {slice_index}
        · {visible_count} associated detection(s) visible
    </div>

    <div style="
        background:#111c2e;
        border:1px solid #263449;
        border-radius:12px;
        padding:12px;
    ">

        <img
            src="data:image/png;base64,{encoded_image}"
            style="
                width:100%;
                height:auto;
                display:block;
                border-radius:8px;
            "
        />

    </div>

    <div style="
        color:#cbd5e1;
        font-size:13px;
        margin-top:12px;
    ">
        <span style="color:#ff4545;">■</span>
        Red rectangle: detected region
        &nbsp;&nbsp;
        <span style="color:#ffdf45;">○</span>
        Yellow circle: estimated centre
    </div>
""")


# ============================================================
# 9. SELECTED CANDIDATE ANALYSIS
# ============================================================

st.divider()

st.header(f"Candidate {selected_group} — Analysis")

score = float(row["Confidence"])

st.html(f"""
    <div style="
        background:#111c2e;
        border:1px solid #263449;
        border-radius:12px;
        padding:20px;
        margin:12px 0;
    ">

        <div style="
            color:#94a3b8;
            font-size:12px;
            letter-spacing:2px;
        ">
            MODEL CONFIDENCE SCORE
        </div>

        <div style="
            color:#38bdf8;
            font-size:36px;
            font-weight:700;
            margin-top:8px;
        ">
            {score:.4f}
        </div>

        <div style="
            color:#94a3b8;
            font-size:13px;
            margin-top:8px;
        ">
            This score describes the model's confidence in its
            detection. It is not a cancer probability or a
            measure of malignancy.
        </div>

    </div>
""")


st.subheader("What does this detection mean?")

st.write(
    str(row["AI_Interpretation"])
)


# ============================================================
# 10. CANDIDATE MEASUREMENTS
# ============================================================

st.subheader("Candidate measurements")

measurement_columns = [
    ("Centre X (mm)", "Center_X_mm"),
    ("Centre Y (mm)", "Center_Y_mm"),
    ("Centre Z (mm)", "Center_Z_mm"),
    ("Width (mm)", "Width_mm"),
    ("Height (mm)", "Height_mm"),
    ("Depth (mm)", "Depth_mm"),
    ("Bounding-box volume (mm³)", "Box_Volume_mm3")
]

measurement_html = ""

for label, column in measurement_columns:

    value = float(row[column])

    measurement_html += f"""
        <tr>
            <td style="
                padding:11px;
                border-bottom:1px solid #263449;
                color:#cbd5e1;
            ">
                {html.escape(label)}
            </td>

            <td style="
                padding:11px;
                border-bottom:1px solid #263449;
                color:#f8fafc;
                text-align:right;
                font-weight:600;
            ">
                {value:.2f}
            </td>
        </tr>
    """


st.html(f"""
    <div style="
        background:#111c2e;
        border:1px solid #263449;
        border-radius:10px;
        overflow:hidden;
    ">

        <table style="
            width:100%;
            border-collapse:collapse;
        ">

            {measurement_html}

        </table>

    </div>

    <p style="
        color:#94a3b8;
        font-size:12px;
        margin-top:10px;
    ">
        Measurement note: These dimensions and volumes describe
        the model's bounding box. They are not verified
        measurements of an actual nodule.
    </p>
""")


# ============================================================
# 11. ALL CANDIDATE GROUPS
# ============================================================

st.divider()

st.header("All Candidate Groups")

st.write(
    "The table below summarizes the "
    f"{len(results)} consolidated candidate groups."
)

table_columns = [
    "Group",
    "Original_Detections",
    "Confidence",
    "Width_mm",
    "Height_mm",
    "Depth_mm"
]

table_html = ""

for _, candidate in results.iterrows():

    table_html += f"""
        <tr>

            <td>{int(candidate["Group"])}</td>

            <td>
                {html.escape(str(candidate["Original_Detections"]))}
            </td>

            <td>
                {float(candidate["Confidence"]):.4f}
            </td>

            <td>
                {float(candidate["Width_mm"]):.2f}
            </td>

            <td>
                {float(candidate["Height_mm"]):.2f}
            </td>

            <td>
                {float(candidate["Depth_mm"]):.2f}
            </td>

        </tr>
    """


st.html(f"""
    <div style="
        overflow-x:auto;
        border:1px solid #263449;
        border-radius:10px;
        background:#111c2e;
    ">

        <table style="
            width:100%;
            border-collapse:collapse;
            color:#e2e8f0;
            font-size:13px;
        ">

            <thead>
                <tr style="
                    background:#1e293b;
                    color:#93c5fd;
                    text-align:left;
                ">

                    <th style="padding:12px;">Candidate</th>
                    <th style="padding:12px;">Original detections</th>
                    <th style="padding:12px;">AI score</th>
                    <th style="padding:12px;">Width (mm)</th>
                    <th style="padding:12px;">Height (mm)</th>
                    <th style="padding:12px;">Depth (mm)</th>

                </tr>
            </thead>

            <tbody>

                {table_html}

            </tbody>

        </table>

    </div>
""")


# ============================================================
# 12. ADDITIONAL CT VISUALIZATIONS
# ============================================================

st.divider()

st.header("Additional CT Visualizations")

display_image_html(
    ORTHOGONAL_PATH,
    "Axial, coronal and sagittal CT views"
)

display_image_html(
    HIGHEST_PATH,
    "Highest-confidence candidate"
)

display_image_html(
    CONFIDENCE_PATH,
    "AI scores across candidate groups"
)

display_image_html(
    GALLERY_PATH,
    "Top candidates (zoomed axial crops)"
)

display_image_html(
    ANALYTICS_PATH,
    "Detection analytics"
)


# ============================================================
# 13. FOOTER
# ============================================================

st.html("""
    <div style="
        text-align:center;
        color:#64748b;
        font-size:12px;
        border-top:1px solid #263449;
        padding-top:22px;
        margin-top:35px;
    ">

        AI LUNG CT ANALYZER · EXPERIMENTAL STUDENT RESEARCH PROJECT

        <br><br>

        Not intended for diagnosis or clinical decision-making.

    </div>
""")