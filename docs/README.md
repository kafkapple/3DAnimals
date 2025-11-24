# 3DAnimals Documentation

**Last Updated**: 2025-11-24

---

## Quick Navigation

### 📚 Research Notes (YYMMDD Format)
Latest research findings and session summaries:

- **[251123 Full Training Session](research/251123_fauna_mouse_full_training_session.md)** - Complete 50K iteration training
- **[251123 Checkpoint Quality Comparison](research/251123_fauna_mouse_checkpoint_quality_comparison.md)** - checkpoint3000 vs checkpoint5000 analysis
- **[251121 System Comprehensive Guide](research/251121_3danimals_system_comprehensive_guide.md)** - End-to-end system overview
- **[251121 Mouse DANNCE Training Setup](research/251121_mouse_dannce_training_setup.md)** - Mouse dataset integration
- **[251119 Mouse Training Setup Session](research/251119_fauna_mouse_training_setup_session.md)** - Initial mouse training configuration
- **[251113 Monocular 3D Reconstruction Analysis](research/251113_monocular_3d_reconstruction_comprehensive_analysis.md)** - Theoretical foundations
- **[251112 Fauna Mouse Final Findings](research/251112_research_fauna_mouse_final_findings.md)** - Mouse integration results
- **[251110 Work Summary: Mammal Integration](research/251110_work_summary_mammal_integration.md)** - Mammal model integration
- **[251110 SDF Pretraining Status](research/251110_mammal_sdf_pretraining_status.md)** - SDF initialization approach
- **[251110 Mouse Prior Shape Integration](research/251110_mammal_mouse_prior_shape_integration.md)** - Prior shape analysis
- **[251110 Mouse Dataset Integration Analysis](research/251110_mouse_dataset_integration_analysis.md)** - Dataset structure analysis
- **[251110 Training & Inference Guide](research/251110_fauna_training_inference_guide.md)** - Training workflow documentation
- **[251109 arti_params None Error](research/20251109_arti_params_none_error.md)** - Bug fix: articulation parameter handling
- **[Inference Quickstart](research/INFERENCE_QUICKSTART.md)** - Quick inference guide

### 📖 Guides
User guides and reference documentation:

#### **⭐ Essential Guides (Start Here)**
- **[Quickstart Manual](guides/QUICKSTART_MANUAL.md)** - **Complete workflow: data → training → inference**
- **[RTX 3060 Setup Guide](guides/RTX_3060_SETUP_GUIDE.md)** - **TF32 CUBLAS error solution**

#### Dataset & Preparation
- **[Fauna Dataset Guide](FAUNA_DATASET_GUIDE.md)** - Dataset structure and usage
- **[Dataset Preparation Guide](FAUNA_DATASET_PREPARATION_GUIDE.md)** - How to prepare datasets

#### Technical Guides
- **[Installation Guide](guides/INSTALL.md)** - Environment setup
- **[Visualization Guide](guides/VISUALIZATION_GUIDE.md)** - Result visualization
- **[CUDA Setup Guide](guides/CUDA_FIX_GUIDE.md)** - CUDA configuration
- **[Architecture Analysis](guides/ARCHITECTURE_ANALYSIS.md)** - System architecture

### 📊 Current Status
Active project status and summaries:

- **[Current Training Status](status/CURRENT_STATUS.md)** - Latest training progress
- **[Latest Session Summary](status/LATEST_SESSION_SUMMARY.md)** - Most recent work summary

### 🗂️ Archive
Historical documents and deprecated content:

- **[Deprecated Files](archive/deprecated/)** - Old status files and outdated approaches

---

## Documentation Organization

This documentation follows a systematic structure:

### Directory Structure

```
docs/
├── README.md                          # This index file
├── FAUNA_DATASET_GUIDE.md             # Main dataset guide
├── FAUNA_DATASET_PREPARATION_GUIDE.md # Dataset preparation
├── research/                          # Research notes (YYMMDD_*.md)
│   ├── 251123_fauna_mouse_full_training_session.md
│   ├── 251123_fauna_mouse_checkpoint_quality_comparison.md
│   └── [other research notes...]
├── status/                            # Current project status
│   ├── CURRENT_STATUS.md
│   └── LATEST_SESSION_SUMMARY.md
└── archive/                           # Historical documents
    ├── deprecated/                    # Outdated files
    ├── old_quickstarts/               # Old quickstart versions
    ├── old_references/                # Old command references
    └── old_guides/                    # Old guide versions
```

### File Naming Conventions

- **Research notes**: `YYMMDD_descriptive_title.md` (e.g., `251123_fauna_mouse_full_training_session.md`)
- **Guides**: `DESCRIPTIVE_TITLE.md` (e.g., `FAUNA_DATASET_GUIDE.md`)
- **Status files**: `CURRENT_STATUS.md`, `LATEST_SESSION_SUMMARY.md`

---

## Key Documentation Files

### For New Users
1. Start with: [251121 System Comprehensive Guide](research/251121_3danimals_system_comprehensive_guide.md)
2. Dataset setup: [Fauna Dataset Guide](FAUNA_DATASET_GUIDE.md)
3. Quick inference: [Inference Quickstart](research/INFERENCE_QUICKSTART.md)

### For Training
1. Training overview: [251121 Mouse DANNCE Training Setup](research/251121_mouse_dannce_training_setup.md)
2. Latest results: [251123 Full Training Session](research/251123_fauna_mouse_full_training_session.md)
3. Quality analysis: [251123 Checkpoint Quality Comparison](research/251123_fauna_mouse_checkpoint_quality_comparison.md)

### For Developers
1. System architecture: [251121 System Comprehensive Guide](research/251121_3danimals_system_comprehensive_guide.md)
2. Bug fixes: [251109 arti_params None Error](research/20251109_arti_params_none_error.md)
3. Dataset integration: [251110 Mouse Dataset Integration Analysis](research/251110_mouse_dataset_integration_analysis.md)

---

## Maintenance Policy

### Adding New Documentation
- **Research notes**: Always use `YYMMDD_topic.md` format in `docs/research/`
- **Status updates**: Update `docs/status/CURRENT_STATUS.md` instead of creating new files
- **Guides**: Update existing guides rather than creating duplicates

### Quarterly Cleanup
- Review `docs/status/` and archive old status files
- Consolidate similar research notes if needed
- Update main guides with latest findings

---

**For questions or issues, refer to the main project README.md in the repository root.**
