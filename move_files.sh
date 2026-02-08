#!/bin/bash
set -x

mkdir -p legacy/prevention_theorem legacy/structural_barriers_preprints legacy/journal_submission_legacy legacy/src_v1_backup legacy/drafts legacy/md_analysis legacy/submission_materials legacy/docs_and_announcements legacy/deprecated_outputs

mv Prevention_Theorem/* legacy/prevention_theorem/ 2>/dev/null
rmdir Prevention_Theorem 2>/dev/null

mv "BMC Public Health/Structural_Barriers_PWIDS_preprints_submission" legacy/structural_barriers_preprints/ 2>/dev/null
mv "BMC Public Health/AIDS_Behavior_Submission" legacy/submission_materials/ 2>/dev/null
mv "BMC Public Health/AIDS_Behvior_submission" legacy/journal_submission_legacy/ 2>/dev/null
mv "BMC Public Health/SRC" legacy/src_v1_backup/ 2>/dev/null
mv "BMC Public Health/AIDS_Behavior_Manuscript.docx" legacy/drafts/ 2>/dev/null
mv AIDS_Behavior_Manuscript_Final.docx legacy/drafts/ 2>/dev/null
mv prevention_theorem_clean-2.bib legacy/prevention_theorem/ 2>/dev/null
mv "BMC Public Health/prevention_theorem_clean-2.bib" legacy/prevention_theorem/ 2>/dev/null

mv docs/* legacy/docs_and_announcements/ 2>/dev/null
rmdir docs 2>/dev/null

mv MD/* legacy/md_analysis/ 2>/dev/null
rmdir MD 2>/dev/null

mv sn-vancouver-num.bst legacy/drafts/ 2>/dev/null

mv "BMC Public Health/Supplemental Data" "Supplemental Data" 2>/dev/null
mv "BMC Public Health/Supplementary Figures" "Supplementary Figures" 2>/dev/null
mv "BMC Public Health/config"/* config/ 2>/dev/null
mv "BMC Public Health/data"/* data/ 2>/dev/null

rm -rf "BMC Public Health"
