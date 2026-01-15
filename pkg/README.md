# pkg 

Conversion of pathology models to torch traces or scripts for simplified interfacing with `bamboo`.

# Setup

To ensure reproducibility, a conda environment is defined for each model. Conversion of each available model is outlined below. All converted models are saved in current directory as [model].traced.pt.

## GrandQC

```bash
(./env/grandqc.sh && python models/grandqc.py && conda deactivate)
```
