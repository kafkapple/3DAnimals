# Documentation Reorganization Plan

**Date**: 2025-11-24
**Purpose**: Streamline documentation, remove duplicates, consolidate information

---

## Current Documentation Analysis

### Root Level Files (26 files)

#### Active/Recent (Keep)
1. `README.md` - Main project documentation ✅
2. `TRAINING_STATUS_UPDATE.md` - Latest training status (Nov 22)
3. `FINAL_SESSION_SUMMARY.md` - Session summary (Nov 22)
4. `VISUALIZATION_GUIDE.md` - How to visualize results (Nov 22)

#### Potentially Outdated/Duplicate
5. `QUICKSTART.md` - General quickstart (Nov 12)
6. `QUICKSTART_NEXT_SESSION.md` - Next session guide (Nov 13)
7. `MOUSE_DANNCE_QUICKSTART.md` - Mouse-specific (Nov 21)
8. `MOUSE_DANNCE_QUICK_START.md` - Duplicate? (Nov 21)
9. `MOUSE_DANNCE_COMMANDS.md` - Command reference (Nov 21)
10. `MOUSE_DANNCE_TRAINING_COMMANDS.md` - Training commands (Nov 21)
11. `MOUSE_DANNCE_TRAINING_GUIDE.md` - Training guide (Nov 21)
12. `FAUNA_TRAINING_QUICK_START.md` - Fauna quickstart (Nov 11)
13. `FAUNA_MOUSE_EXECUTION_PLAN.md` - Execution plan (Nov 13)

#### Deprecated/Historical
14. `STATUS_BLOCKED_NUM_ITERS.md` - Old status (Nov 11)
15. `STATUS_CURRENT_SESSION.md` - Old status (Nov 13)
16. `MOUSE_ONLY_TRAINING_SETUP.md` - Failed approach (Nov 22)
17. `TROUBLESHOOTING_SESSION_251122.md` - Specific session (Nov 22)
18. `INFERENCE_RESULTS_COMPARISON.md` - Old comparison (Nov 22)
19. `FAUNA_ISSUES_LOG.md` - Old issues (Nov 11)

#### Technical/Reference
20. `CUDA_FIX_GUIDE.md` - CUDA setup (Nov 21)
21. `FIX_CUDA_FINAL.md` - CUDA fix (Nov 21)
22. `POSE_SPLATTER_INTEGRATION_GUIDE.md` - Integration guide (Nov 9)
23. `ARCHITECTURE_ANALYSIS.md` - Code analysis (Nov 9)
24. `ANALYSIS_INDEX.md` - Analysis index (Nov 9)
25. `QUICK_REFERENCE.md` - Quick reference (Nov 9)
26. `INSTALL.md` - Installation guide (Nov 9)

---

### docs/ Directory Structure

```
docs/
├── README.md                                    # Index
├── FAUNA_DATASET_GUIDE.md                       # Dataset guide (Nov 19)
├── FAUNA_DATASET_PREPARATION_GUIDE.md           # Dataset prep (Nov 21)
├── 251112_research_fauna_mouse_final_findings.md # Research note
├── reports/
│   ├── 20251109_arti_params_none_error.md
│   ├── 251110_fauna_training_inference_guide.md
│   ├── 251110_mammal_mouse_prior_shape_integration.md
│   ├── 251110_mammal_sdf_pretraining_status.md
│   ├── 251110_mouse_dataset_integration_analysis.md
│   ├── 251110_work_summary_mammal_integration.md
│   ├── 251113_monocular_3d_reconstruction_comprehensive_analysis.md
│   ├── 251119_fauna_mouse_training_setup_session.md
│   ├── 251121_3danimals_system_comprehensive_guide.md
│   ├── 251121_mouse_dannce_training_setup.md
│   ├── 251123_fauna_mouse_checkpoint_quality_comparison.md  # LATEST
│   ├── 251123_fauna_mouse_full_training_session.md          # LATEST
│   └── INFERENCE_QUICKSTART.md
├── guides/
│   └── (empty?)
└── troubleshooting/
    └── (empty?)
```

---

## Document Categories

### 1. User Documentation (For End Users)
**Purpose**: Help users use the system

- README.md (main entry point)
- Installation guides
- Quickstart guides
- Command references
- Troubleshooting

### 2. Research Notes (Historical Record)
**Purpose**: Document experiments, findings, decisions

- Dated research reports (YYMMDD_*.md)
- Session summaries
- Experiment results

### 3. Technical Reference (Developer Docs)
**Purpose**: Understand system internals

- Architecture analysis
- Code structure
- Integration guides

### 4. Status/Progress (Ephemeral)
**Purpose**: Track current work

- Training status
- Current session notes
- Action plans

---

## Reorganization Plan

### Phase 1: Archive Deprecated Files

**Move to `docs/archive/deprecated/`**:
```bash
mkdir -p docs/archive/deprecated

# Outdated status files
mv STATUS_BLOCKED_NUM_ITERS.md docs/archive/deprecated/
mv STATUS_CURRENT_SESSION.md docs/archive/deprecated/
mv FAUNA_ISSUES_LOG.md docs/archive/deprecated/

# Failed approaches
mv MOUSE_ONLY_TRAINING_SETUP.md docs/archive/deprecated/
mv TROUBLESHOOTING_SESSION_251122.md docs/archive/deprecated/
mv INFERENCE_RESULTS_COMPARISON.md docs/archive/deprecated/

# Old execution plans
mv FAUNA_MOUSE_EXECUTION_PLAN.md docs/archive/deprecated/
```

### Phase 2: Consolidate Quickstart Guides

**Create single comprehensive quickstart** → `docs/QUICKSTART.md`:

Merge content from:
- QUICKSTART.md
- QUICKSTART_NEXT_SESSION.md
- MOUSE_DANNCE_QUICKSTART.md
- MOUSE_DANNCE_QUICK_START.md
- FAUNA_TRAINING_QUICK_START.md

**Then archive originals** → `docs/archive/old_quickstarts/`

### Phase 3: Consolidate Command References

**Create single command reference** → `docs/COMMAND_REFERENCE.md`:

Merge content from:
- MOUSE_DANNCE_COMMANDS.md
- MOUSE_DANNCE_TRAINING_COMMANDS.md
- QUICK_REFERENCE.md

**Then archive originals** → `docs/archive/old_references/`

### Phase 4: Organize Training Guides

**Create consolidated training guide** → `docs/TRAINING_GUIDE.md`:

Merge content from:
- MOUSE_DANNCE_TRAINING_GUIDE.md
- Latest research findings

**Then archive originals** → `docs/archive/old_guides/`

### Phase 5: Organize by Category

**Final structure**:
```
/home/joon/dev/3DAnimals/
├── README.md                          # Main entry (keep at root)
│
├── docs/
│   ├── README.md                      # Documentation index
│   │
│   ├── quickstart/
│   │   ├── QUICKSTART.md              # Consolidated quickstart
│   │   ├── INSTALLATION.md            # From INSTALL.md
│   │   └── VISUALIZATION.md           # From VISUALIZATION_GUIDE.md
│   │
│   ├── guides/
│   │   ├── TRAINING_GUIDE.md          # Consolidated training
│   │   ├── DATASET_GUIDE.md           # Fauna dataset
│   │   ├── COMMAND_REFERENCE.md       # All commands
│   │   └── CUDA_SETUP.md              # From CUDA_FIX_GUIDE.md
│   │
│   ├── reference/
│   │   ├── ARCHITECTURE.md            # System architecture
│   │   └── INTEGRATION.md             # Pose-splatter integration
│   │
│   ├── research/
│   │   ├── 251112_fauna_mouse_final_findings.md
│   │   ├── 251123_fauna_mouse_checkpoint_quality_comparison.md
│   │   ├── 251123_fauna_mouse_full_training_session.md
│   │   └── (other research notes - dated YYMMDD_*.md)
│   │
│   ├── status/
│   │   ├── CURRENT_STATUS.md          # Latest status
│   │   └── TRAINING_LOG.md            # Training history
│   │
│   └── archive/
│       ├── deprecated/                # Outdated files
│       ├── old_quickstarts/          # Old quickstart versions
│       ├── old_references/           # Old command refs
│       └── old_guides/               # Old guide versions
│
└── (project files)
```

---

## Consolidation Details

### Quickstart Guide Structure

**docs/quickstart/QUICKSTART.md**:
```markdown
# 3DAnimals Quickstart Guide

## Prerequisites
- CUDA 11.8
- Python 3.9
- Conda environment

## Installation
[From INSTALL.md]

## Quick Start (5 minutes)
1. Setup environment
2. Download dataset
3. Run inference

## Training (Fauna Mouse)
[From MOUSE_DANNCE_QUICKSTART.md]
- Debug mode (30 min)
- Full training (2-3 hours)

## Visualization
[From VISUALIZATION_GUIDE.md]
- View results
- Compare checkpoints

## Next Steps
- Advanced training
- Custom datasets
```

### Command Reference Structure

**docs/guides/COMMAND_REFERENCE.md**:
```markdown
# Command Reference

## Training Commands
### Debug Training
### Full Training
### Resume Training

## Inference Commands
### Standard Inference
### Batch Inference
### Custom Config

## Monitoring Commands
### GPU Usage
### Training Progress
### Checkpoint Management

## Troubleshooting Commands
### Check Environment
### Verify Dataset
### Debug Errors
```

### Training Guide Structure

**docs/guides/TRAINING_GUIDE.md**:
```markdown
# Training Guide

## Overview
- Multi-animal vs Single-animal
- Progressive training concept
- Hardware requirements

## Dataset Preparation
[From FAUNA_DATASET_GUIDE.md]

## Training Workflow
1. Debug-first principle
2. Full training
3. Checkpoints and evaluation

## Mouse-Specific Training
[From MOUSE_DANNCE_TRAINING_GUIDE.md]
- Few-shot challenges
- Multi-animal strategy
- Expected results

## Troubleshooting
- Common errors
- Solutions

## Best Practices
[From latest research notes]
```

---

## Implementation Steps

### Step 1: Create Directory Structure
```bash
cd /home/joon/dev/3DAnimals/docs

mkdir -p quickstart
mkdir -p guides
mkdir -p reference
mkdir -p research
mkdir -p status
mkdir -p archive/{deprecated,old_quickstarts,old_references,old_guides}
```

### Step 2: Move Files to Archive
```bash
# See Phase 1 above
```

### Step 3: Create Consolidated Documents
```bash
# Quickstart
cat ../QUICKSTART.md ../QUICKSTART_NEXT_SESSION.md ../MOUSE_DANNCE_QUICKSTART.md \
  > quickstart/QUICKSTART.md.draft

# Command Reference
cat ../MOUSE_DANNCE_COMMANDS.md ../MOUSE_DANNCE_TRAINING_COMMANDS.md ../QUICK_REFERENCE.md \
  > guides/COMMAND_REFERENCE.md.draft

# Training Guide
cat ../MOUSE_DANNCE_TRAINING_GUIDE.md \
  > guides/TRAINING_GUIDE.md.draft

# Manual editing required to remove duplicates and organize
```

### Step 4: Move Supporting Files
```bash
# Move to appropriate categories
mv ../INSTALL.md quickstart/INSTALLATION.md
mv ../VISUALIZATION_GUIDE.md quickstart/VISUALIZATION.md
mv ../CUDA_FIX_GUIDE.md guides/CUDA_SETUP.md
mv ../ARCHITECTURE_ANALYSIS.md reference/ARCHITECTURE.md
mv ../POSE_SPLATTER_INTEGRATION_GUIDE.md reference/INTEGRATION.md

# Move research notes
mv ../docs/reports/25*.md research/
mv ../docs/251112_research_fauna_mouse_final_findings.md research/

# Current status
cp ../TRAINING_STATUS_UPDATE.md status/CURRENT_STATUS.md
cp ../FINAL_SESSION_SUMMARY.md status/LATEST_SESSION_SUMMARY.md
```

### Step 5: Create Index
**docs/README.md**:
```markdown
# 3DAnimals Documentation

## Quick Start
- [Quickstart Guide](quickstart/QUICKSTART.md)
- [Installation](quickstart/INSTALLATION.md)
- [Visualization](quickstart/VISUALIZATION.md)

## Guides
- [Training Guide](guides/TRAINING_GUIDE.md)
- [Dataset Guide](guides/DATASET_GUIDE.md)
- [Command Reference](guides/COMMAND_REFERENCE.md)
- [CUDA Setup](guides/CUDA_SETUP.md)

## Technical Reference
- [System Architecture](reference/ARCHITECTURE.md)
- [Integration Guide](reference/INTEGRATION.md)

## Research Notes
- [Latest Session (Nov 23)](research/251123_fauna_mouse_full_training_session.md)
- [Checkpoint Comparison](research/251123_fauna_mouse_checkpoint_quality_comparison.md)
- [All Research Notes](research/)

## Current Status
- [Training Status](status/CURRENT_STATUS.md)
- [Latest Summary](status/LATEST_SESSION_SUMMARY.md)
```

---

## Priority Actions

### High Priority (Do Now)
1. ✅ Create directory structure
2. ✅ Archive deprecated files
3. ✅ Create consolidated QUICKSTART.md
4. ✅ Create COMMAND_REFERENCE.md
5. ✅ Update docs/README.md index

### Medium Priority (This Week)
6. Consolidate training guides
7. Organize research notes by topic
8. Clean up root directory

### Low Priority (As Needed)
9. Update integration guides
10. Expand troubleshooting
11. Add examples

---

## File Disposition

### Keep at Root
- README.md (main project entry)

### Keep in docs/
- All organized documentation

### Delete
- None (archive instead for history)

### Archive
- See Phase 1-4 above

---

## Maintenance Policy

### Going Forward

1. **Research Notes**: Always use `YYMMDD_topic.md` format in `docs/research/`
2. **Status Updates**: Update `docs/status/CURRENT_STATUS.md`, don't create new files
3. **Guides**: Update existing guides, don't duplicate
4. **Quick Fixes**: Add to `docs/guides/TROUBLESHOOTING.md`

### Quarterly Cleanup
- Review `docs/status/` and archive old status
- Consolidate similar research notes
- Update main guides with latest findings

---

## Success Criteria

✅ **User can find information quickly**
- 3 clicks max from docs/README.md to any info
- Clear navigation
- No duplicates

✅ **Historical record preserved**
- All research notes archived
- Progression visible

✅ **Maintenance is easy**
- Clear policy
- Consistent naming
- Logical structure

---

**Next Action**: Execute Step 1-5 systematically
