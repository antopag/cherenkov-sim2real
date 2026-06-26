# paper_electronics/

Deliverables for the MDPI Electronics paper on carrier-DA interaction.

## Regenerating figures

```bash
cd cherenkov-sim2real

# 1. Extract consolidated results from MLflow runs
python paper_electronics/extract_paper_data.py

# 2. Generate all figures
cd paper_electronics/figures
python generate_fig1_schematic.py
python generate_fig2_carrier_gradient.py
python generate_fig3_mm_vs_coral.py
python generate_fig4_shift_diagnostics.py
python generate_fig5_threshold_schematic.py
```

Outputs: `paper_electronics/figures/output/*.{pdf,png}`

## Figure inventory

| Figure | Type | Width | Description |
|--------|------|-------|-------------|
| Fig 1 | Conceptual | Single-col | 2x2 framework schematic |
| Fig 2 | Data | Double-col | Carrier gradient (main result) |
| Fig 3 | Data | Single-col | Mean matching vs CORAL on tree carriers |
| Fig 4 | Data | Single-col | Shift diagnostics + carrier correlation |
| Fig 5 | Conceptual | Single-col | Threshold-split mechanism |

## Data source

All numbers derive from S0 experiments (sessions 5-8) stored in
`results/_mlruns/` and consolidated into
`paper_electronics/data/extracted_results.json`.
