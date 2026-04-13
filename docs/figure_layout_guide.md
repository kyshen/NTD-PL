# Figure Layout Guide

This file records the intended paper-layout role of each generated figure.

| Figure ID | Output | Role | Suggested LaTeX insertion | Shared legend | Pairing advice | Appendix only |
| --- | --- | --- | --- | --- | --- | --- |
| `linear_consistency_paired_gap` | `experiment/outputs/figures/main/linear_consistency_paired_gap.pdf` | single-column | `\includegraphics[width=\linewidth]{./inputs/figures/main/linear_consistency_paired_gap.pdf}` | no | standalone | no |
| `nonlinear_pmax_grid` | `experiment/outputs/figures/main/nonlinear_pmax_grid.pdf` | double-column | `\includegraphics[width=\textwidth]{./inputs/figures/main/nonlinear_pmax_grid.pdf}` | yes | keep adjacent to `nonlinear_alpha_grid` | no |
| `nonlinear_alpha_grid` | `experiment/outputs/figures/main/nonlinear_alpha_grid.pdf` | double-column | `\includegraphics[width=\textwidth]{./inputs/figures/main/nonlinear_alpha_grid.pdf}` | yes | keep adjacent to `nonlinear_pmax_grid` | no |
| `nonlinear_step_grid` | `experiment/outputs/figures/main/nonlinear_step_grid.pdf` | double-column | `\includegraphics[width=\textwidth]{./inputs/figures/main/nonlinear_step_grid.pdf}` | yes | standalone | no |
| `cave_reconstruction_scene_gain` | `experiment/outputs/figures/main/cave_reconstruction_scene_gain.pdf` | single-column | `\includegraphics[width=\linewidth]{./inputs/figures/main/cave_reconstruction_scene_gain.pdf}` | no | standalone | no |
| `cave_reconstruction_visual_grid` | `experiment/outputs/figures/main/cave_reconstruction_visual_grid.pdf` | double-column | `\includegraphics[width=\textwidth]{./inputs/figures/main/cave_reconstruction_visual_grid.pdf}` | colorbar only | do not place side-by-side | no |
| `cave_reconstruction_spectra` | `experiment/outputs/figures/main/cave_reconstruction_spectra.pdf` | double-column | `\includegraphics[width=\textwidth]{./inputs/figures/main/cave_reconstruction_spectra.pdf}` | yes | standalone | no |
| `cave_completion_scene_gain` | `experiment/outputs/figures/main/cave_completion_scene_gain.pdf` | double-column | `\includegraphics[width=\textwidth]{./inputs/figures/main/cave_completion_scene_gain.pdf}` | no | standalone | no |
| `cave_completion_advantage_heatmap` | `experiment/outputs/figures/main/cave_completion_advantage_heatmap.pdf` | single-column | `\includegraphics[width=\linewidth]{./inputs/figures/main/cave_completion_advantage_heatmap.pdf}` | no | standalone | no |
| `nonlinear_beta_distribution` | `experiment/outputs/figures/appendix/nonlinear_beta_distribution.pdf` | appendix-wide | `\includegraphics[width=\textwidth]{./inputs/figures/appendix/nonlinear_beta_distribution.pdf}` | no | appendix only | yes |
| `geometry_link_evolution` | `experiment/outputs/figures/appendix/geometry_link_evolution.pdf` | appendix-wide | `\includegraphics[width=\textwidth]{./inputs/figures/appendix/geometry_link_evolution.pdf}` | yes | candidate for promotion to main text if a mechanism figure is needed | no |
| `geometry_response_maps` | `experiment/outputs/figures/appendix/geometry_response_maps.pdf` | appendix-wide | `\includegraphics[width=\textwidth]{./inputs/figures/appendix/geometry_response_maps.pdf}` | shared color scales | keep in appendix, adjacent to `geometry_link_evolution` | yes |

## Notes

- Main-text figures are intentionally fewer and larger than the legacy outputs.
- Multi-panel figures use their own role-specific height presets; they should not be further shrunk with `resizebox`.
- Shared legends are reserved for multi-curve plots only. The image and heatmap figures avoid repeated per-panel legends.
- The appendix figures keep higher information density but still use fixed-width presets rather than ad hoc scaling.
