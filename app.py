# ============================================================
# app.py
# Human Brain Vascular snRNA-seq Streamlit Application
# ============================================================

import streamlit as st

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tarfile
import zipfile
import tempfile
import shutil
from pathlib import Path


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Human Brain Vascular snRNA-seq Analysis",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🧠 Human Brain Vascular snRNA-seq Analysis")

st.markdown(
    """
    Analyze a processed **10x Genomics single-nucleus RNA-seq sample**
    using vascular gene modules derived from the GSE163577 analysis.
    """
)

st.info(
    """
    Upload one `.tar.gz`, `.tgz`, `.tar`, or `.zip` archive containing
    the standard 10x Genomics files:

    `matrix.mtx.gz`  
    `barcodes.tsv.gz`  
    `features.tsv.gz`
    """
)


# ============================================================
# GENE MODULES
# ============================================================

MODULES = {

    "Junction": [
        "CLDN5",
        "OCLN",
        "TJP1",
        "CDH5",
        "JAM2",
        "JAM3"
    ],

    "Cytoskeleton": [
        "ACTB",
        "ACTN1",
        "MYH9",
        "PFN1",
        "ARPC2",
        "RHOA"
    ],

    "Transport": [
        "SLC2A1",
        "ABCB1",
        "SLC7A5",
        "MFSD2A",
        "SLC16A1",
        "ABCG2"
    ]
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Application")

st.sidebar.markdown(
    """
    **Input**

    One processed 10x Genomics sample.

    The archive may contain nested folders. The application
    automatically searches for:

    - `matrix.mtx.gz`
    - `barcodes.tsv.gz`
    - `features.tsv.gz`
    """
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Analysis")

st.sidebar.markdown(
    """
    1. Load 10x matrix
    2. Basic QC
    3. Normalize expression
    4. Calculate vascular module scores
    5. Visualize module scores
    6. Download results
    """
)


# ============================================================
# UTILITY — FIND 10X DIRECTORY
# ============================================================

def find_10x_directory(root_dir):
    """
    Recursively search for the directory containing:

        matrix.mtx.gz
        barcodes.tsv.gz
        features.tsv.gz

    This handles structures such as:

    GSM.../
        01_05_AD_filtered_feature_bc_matrix/
            matrix.mtx.gz
            barcodes.tsv.gz
            features.tsv.gz
    """

    root_dir = Path(root_dir)

    # Search root itself and all subdirectories
    directories = [root_dir]

    directories.extend(
        [
            p
            for p in root_dir.rglob("*")
            if p.is_dir()
        ]
    )

    for directory in directories:

        matrix_file = directory / "matrix.mtx.gz"
        barcodes_file = directory / "barcodes.tsv.gz"
        features_file = directory / "features.tsv.gz"

        if (
            matrix_file.exists()
            and barcodes_file.exists()
            and features_file.exists()
        ):
            return directory

        # Also support uncompressed files
        matrix_file = directory / "matrix.mtx"
        barcodes_file = directory / "barcodes.tsv"
        features_file = directory / "features.tsv"

        if (
            matrix_file.exists()
            and barcodes_file.exists()
            and features_file.exists()
        ):
            return directory

    return None


# ============================================================
# UTILITY — EXTRACT ARCHIVE
# ============================================================

def extract_archive(uploaded_file, temp_dir):

    filename = uploaded_file.name.lower()

    archive_path = (
        Path(temp_dir)
        / uploaded_file.name
    )

    # Save uploaded file temporarily
    with open(archive_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    extract_dir = (
        Path(temp_dir)
        / "extracted"
    )

    extract_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    if filename.endswith(".zip"):

        with zipfile.ZipFile(
            archive_path,
            "r"
        ) as archive:

            archive.extractall(
                extract_dir
            )

    # --------------------------------------------------------
    # TAR / TAR.GZ / TGZ
    # --------------------------------------------------------

    elif (
        filename.endswith(".tar.gz")
        or filename.endswith(".tgz")
        or filename.endswith(".tar")
    ):

        with tarfile.open(
            archive_path,
            "r:*"
        ) as archive:

            archive.extractall(
                extract_dir
            )

    else:

        raise ValueError(
            "Unsupported file format. "
            "Please upload a .tar.gz, .tgz, .tar, or .zip file."
        )

    return extract_dir


# ============================================================
# UTILITY — LOAD 10X DATA
# ============================================================

def load_10x_sample(matrix_dir):

    """
    Load the filtered feature-barcode matrix using Scanpy.
    """

    matrix_dir = Path(matrix_dir)

    adata = sc.read_10x_mtx(
        matrix_dir,
        var_names="gene_symbols",
        make_unique=True
    )

    return adata


# ============================================================
# UTILITY — QC
# ============================================================

def calculate_qc_metrics(adata):

    """
    Calculate basic QC metrics.

    These metrics are displayed to the user.
    """

    # Identify mitochondrial genes
    adata.var["mt"] = (
        adata.var_names
        .str.upper()
        .str.startswith("MT-")
    )

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        inplace=True
    )

    return adata


# ============================================================
# UTILITY — NORMALIZATION
# ============================================================

def normalize_for_scoring(adata):

    """
    Normalize expression before module scoring.

    Important:
    The raw count matrix is preserved in adata.layers["counts"].
    """

    # Preserve raw counts
    adata.layers["counts"] = (
        adata.X.copy()
    )

    # Normalize each nucleus
    sc.pp.normalize_total(
        adata,
        target_sum=1e4
    )

    # Log transform
    sc.pp.log1p(
        adata
    )

    return adata


# ============================================================
# UTILITY — MODULE SCORING
# ============================================================

def calculate_module_scores(adata):

    """
    Calculate Junction, Cytoskeleton and Transport
    module scores.

    Genes not present in the uploaded sample are reported.
    """

    results = {}

    for module_name, genes in MODULES.items():

        # ----------------------------------------------------
        # Determine which genes are available
        # ----------------------------------------------------

        available_genes = [
            gene
            for gene in genes
            if gene in adata.var_names
        ]

        missing_genes = [
            gene
            for gene in genes
            if gene not in adata.var_names
        ]

        score_name = (
            module_name.lower()
            + "_score"
        )

        # ----------------------------------------------------
        # No genes available
        # ----------------------------------------------------

        if len(available_genes) == 0:

            results[module_name] = {
                "score_column": score_name,
                "mean_score": np.nan,
                "median_score": np.nan,
                "std_score": np.nan,
                "available_genes": [],
                "missing_genes": missing_genes
            }

            continue

        # ----------------------------------------------------
        # Calculate Scanpy module score
        # ----------------------------------------------------

        sc.tl.score_genes(
            adata,
            gene_list=available_genes,
            score_name=score_name,
            use_raw=False
        )

        score_values = (
            adata.obs[score_name]
            .astype(float)
        )

        results[module_name] = {

            "score_column": score_name,

            "mean_score": float(
                score_values.mean()
            ),

            "median_score": float(
                score_values.median()
            ),

            "std_score": float(
                score_values.std()
            ),

            "available_genes": available_genes,

            "missing_genes": missing_genes
        }

    return results


# ============================================================
# UTILITY — RESULTS TABLE
# ============================================================

def create_results_dataframe(results):

    rows = []

    for module_name, result in results.items():

        rows.append({

            "Module":
                module_name,

            "Mean Score":
                result["mean_score"],

            "Median Score":
                result["median_score"],

            "Std":
                result["std_score"],

            "Genes Found":
                len(
                    result["available_genes"]
                ),

            "Genes Expected":
                len(
                    MODULES[module_name]
                ),

            "Genes Missing":
                len(
                    result["missing_genes"]
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# UTILITY — SAMPLE INFORMATION
# ============================================================

def infer_sample_information(filename):

    """
    Extract simple information from the filename.

    This is ONLY displayed as filename-derived metadata.
    It is NOT used to make an AD prediction.
    """

    name = filename.lower()

    information = {}

    if "_ad_" in name:

        information["Disease label in filename"] = "AD"

    elif "_c_" in name:

        information["Disease label in filename"] = "Control"

    else:

        information["Disease label in filename"] = (
            "Not identified"
        )

    return information


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "📁 Upload your 10x Genomics sample",

    type=[
        "tar.gz",
        "tgz",
        "tar",
        "zip"
    ],

    help=(
        "Upload an archive containing "
        "matrix.mtx.gz, barcodes.tsv.gz, "
        "and features.tsv.gz."
    )
)


# ============================================================
# WAIT FOR UPLOAD
# ============================================================

if uploaded_file is None:

    st.markdown(
        "## Expected input structure"
    )

    st.code(
        """
GSM4982084_01_05_AD_filtered_feature_bc_matrix.tar.gz
│
└── GSM4982084_01_05_AD_filtered_feature_bc_matrix/
    │
    └── 01_05_AD_filtered_feature_bc_matrix/
        ├── barcodes.tsv.gz
        ├── features.tsv.gz
        └── matrix.mtx.gz
        """
    )

    st.markdown(
        """
        You can upload the **whole `.tar.gz` archive**.
        You do not need to extract it manually.
        """
    )

    st.stop()


# ============================================================
# UPLOAD INFORMATION
# ============================================================

st.success(
    f"Uploaded: **{uploaded_file.name}**"
)

file_size_mb = (
    uploaded_file.size
    / (1024 * 1024)
)

st.write(
    f"File size: `{file_size_mb:.2f} MB`"
)


# ============================================================
# SAMPLE METADATA
# ============================================================

sample_information = (
    infer_sample_information(
        uploaded_file.name
    )
)

with st.expander(
    "📋 Sample information"
):

    for key, value in (
        sample_information.items()
    ):

        st.write(
            f"**{key}:** {value}"
        )

    st.caption(
        """
        Filename-derived disease information is shown only as
        metadata. It is not used by the analysis to classify
        the uploaded sample.
        """
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔬 Analyze Sample",
    type="primary",
    use_container_width=True
)


if analyze:

    # ========================================================
    # TEMPORARY DIRECTORY
    # ========================================================

    temp_dir = tempfile.mkdtemp()

    try:

        # ====================================================
        # STEP 1 — EXTRACT
        # ====================================================

        st.header(
            "1. Loading sample"
        )

        progress = st.progress(
            0
        )

        status = st.empty()

        status.write(
            "Extracting uploaded archive..."
        )

        extracted_dir = (
            extract_archive(
                uploaded_file,
                temp_dir
            )
        )

        progress.progress(
            20
        )

        # ====================================================
        # STEP 2 — FIND MATRIX
        # ====================================================

        status.write(
            "Searching for the 10x feature-barcode matrix..."
        )

        matrix_dir = (
            find_10x_directory(
                extracted_dir
            )
        )

        if matrix_dir is None:

            progress.empty()

            st.error(
                """
                ❌ No valid 10x Genomics matrix was found.

                The uploaded archive must contain:

                matrix.mtx.gz
                barcodes.tsv.gz
                features.tsv.gz
                """
            )

            st.stop()

        st.success(
            f"""
            10x matrix found:

            `{matrix_dir}`
            """
        )

        progress.progress(
            30
        )

        # ====================================================
        # STEP 3 — LOAD ANNData
        # ====================================================

        status.write(
            "Reading expression matrix with Scanpy..."
        )

        adata = (
            load_10x_sample(
                matrix_dir
            )
        )

        progress.progress(
            45
        )

        # ====================================================
        # BASIC DATASET INFORMATION
        # ====================================================

        st.header(
            "2. Dataset information"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        col1.metric(
            "Nuclei",
            f"{adata.n_obs:,}"
        )

        col2.metric(
            "Genes",
            f"{adata.n_vars:,}"
        )

        col3.metric(
            "Matrix size",
            f"{adata.n_obs:,} × {adata.n_vars:,}"
        )

        # ====================================================
        # STEP 4 — QC
        # ====================================================

        status.write(
            "Calculating quality-control metrics..."
        )

        adata = (
            calculate_qc_metrics(
                adata
            )
        )

        progress.progress(
            55
        )

        st.header(
            "3. Quality control"
        )

        # ----------------------------------------------------
        # QC SUMMARY
        # ----------------------------------------------------

        qc_col1, qc_col2, qc_col3, qc_col4 = (
            st.columns(4)
        )

        qc_col1.metric(
            "Median genes / nucleus",
            f"{adata.obs['n_genes_by_counts'].median():.0f}"
        )

        qc_col2.metric(
            "Median UMI / nucleus",
            f"{adata.obs['total_counts'].median():.0f}"
        )

        qc_col3.metric(
            "Median mitochondrial %",
            f"{adata.obs['pct_counts_mt'].median():.2f}%"
        )

        qc_col4.metric(
            "Nuclei",
            f"{adata.n_obs:,}"
        )

        # ----------------------------------------------------
        # QC PLOTS
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 4)
        )

        axes[0].hist(
            adata.obs[
                "n_genes_by_counts"
            ],
            bins=50
        )

        axes[0].set_title(
            "Genes per nucleus"
        )

        axes[0].set_xlabel(
            "Genes"
        )

        axes[0].set_ylabel(
            "Nuclei"
        )

        axes[1].hist(
            adata.obs[
                "total_counts"
            ],
            bins=50
        )

        axes[1].set_title(
            "UMI counts per nucleus"
        )

        axes[1].set_xlabel(
            "UMI counts"
        )

        axes[1].set_ylabel(
            "Nuclei"
        )

        axes[2].hist(
            adata.obs[
                "pct_counts_mt"
            ],
            bins=50
        )

        axes[2].set_title(
            "Mitochondrial percentage"
        )

        axes[2].set_xlabel(
            "Mitochondrial %"
        )

        axes[2].set_ylabel(
            "Nuclei"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )

        # ====================================================
        # STEP 5 — NORMALIZATION
        # ====================================================

        status.write(
            "Normalizing expression data..."
        )

        adata = (
            normalize_for_scoring(
                adata
            )
        )

        progress.progress(
            65
        )

        # ====================================================
        # STEP 6 — MODULE SCORING
        # ====================================================

        status.write(
            "Calculating vascular module scores..."
        )

        results = (
            calculate_module_scores(
                adata
            )
        )

        progress.progress(
            80
        )

        st.header(
            "4. Vascular module scores"
        )

        st.markdown(
            """
            The uploaded sample is evaluated using three predefined
            vascular gene modules:
            """
        )

        # ----------------------------------------------------
        # RESULTS TABLE
        # ----------------------------------------------------

        results_df = (
            create_results_dataframe(
                results
            )
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # STEP 7 — BAR CHART
        # ====================================================

        st.subheader(
            "Module score summary"
        )

        valid_results = (
            results_df[
                results_df["Mean Score"].notna()
            ]
        )

        if len(valid_results) > 0:

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            ax.bar(
                valid_results["Module"],
                valid_results["Mean Score"]
            )

            ax.set_ylabel(
                "Mean module score"
            )

            ax.set_xlabel(
                "Module"
            )

            ax.set_title(
                "Mean vascular module scores"
            )

            ax.axhline(
                0,
                linestyle="--",
                linewidth=1
            )

            plt.tight_layout()

            st.pyplot(
                fig
            )

            plt.close(
                fig
            )

        # ====================================================
        # STEP 8 — DISTRIBUTIONS
        # ====================================================

        st.subheader(
            "Module score distributions"
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16, 4)
        )

        for ax, module_name in zip(
            axes,
            MODULES.keys()
        ):

            score_column = (
                results[
                    module_name
                ]["score_column"]
            )

            if score_column in adata.obs:

                values = (
                    adata.obs[
                        score_column
                    ]
                    .astype(float)
                    .values
                )

                ax.hist(
                    values,
                    bins=40
                )

                ax.axvline(
                    np.mean(values),
                    linestyle="--",
                    linewidth=1
                )

            ax.set_title(
                module_name
            )

            ax.set_xlabel(
                "Module score"
            )

            ax.set_ylabel(
                "Nuclei"
            )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(
            fig
        )

        progress.progress(
            90
        )

        # ====================================================
        # STEP 9 — GENE COVERAGE
        # ====================================================

        st.header(
            "5. Gene coverage"
        )

        for module_name, result in (
            results.items()
        ):

            with st.expander(
                f"{module_name} module"
            ):

                col1, col2 = (
                    st.columns(2)
                )

                with col1:

                    st.markdown(
                        "**Genes found**"
                    )

                    if result[
                        "available_genes"
                    ]:

                        st.write(
                            ", ".join(
                                result[
                                    "available_genes"
                                ]
                            )
                        )

                    else:

                        st.write(
                            "None"
                        )

                with col2:

                    st.markdown(
                        "**Genes missing**"
                    )

                    if result[
                        "missing_genes"
                    ]:

                        st.warning(
                            ", ".join(
                                result[
                                    "missing_genes"
                                ]
                            )
                        )

                    else:

                        st.success(
                            "None"
                        )

        # ====================================================
        # STEP 10 — DOWNLOAD SUMMARY
        # ====================================================

        st.header(
            "6. Download results"
        )

        summary_csv = (
            results_df.to_csv(
                index=False
            )
        )

        st.download_button(
            label="⬇️ Download module-score summary",
            data=summary_csv,
            file_name=(
                "sample_module_scores.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

        # ====================================================
        # OPTIONAL — PER-NUCLEUS SCORES
        # ====================================================

        st.subheader(
            "Per-nucleus scores"
        )

        per_cell_columns = []

        for module_name in MODULES:

            column = (
                results[
                    module_name
                ]["score_column"]
            )

            if column in adata.obs:

                per_cell_columns.append(
                    column
                )

        if per_cell_columns:

            per_cell_df = (
                adata.obs[
                    per_cell_columns
                ]
                .copy()
            )

            per_cell_df.index.name = (
                "Nucleus"
            )

            st.dataframe(
                per_cell_df.head(100),
                use_container_width=True
            )

            per_cell_csv = (
                per_cell_df.to_csv()
            )

            st.download_button(
                label=(
                    "⬇️ Download per-nucleus scores"
                ),
                data=per_cell_csv,
                file_name=(
                    "per_nucleus_module_scores.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

        # ====================================================
        # FINISH
        # ====================================================

        progress.progress(
            100
        )

        status.success(
            "Analysis completed successfully."
        )

        st.success(
            """
            ✅ Sample analysis completed.

            The displayed module scores describe the vascular
            expression patterns in the uploaded sample. They are
            not an independent AD diagnosis.
            """
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            "❌ An error occurred while processing the sample."
        )

        st.exception(
            e
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )