# HET: Heterogeneity of Multiple System Atrophy Using Machine Learning and MRI

This repository contains the pipeline for running several machine learning models with MRI-derived features as inputs and diagnosis as output: multiple system atrophy (0) and Parkinson's disease (PD, 1). Cross-validation using ten random seeds is used to search for the optimal model with least overfitting and best performance. The final model is used to compute **Heterogeneity (HET) scores** to account for subtype specific heterogeneity in MSA. The HET framework captures spatial heterogeneity across structural and diffusion MRI inputs and is derived directly from model-based feature importances which avoids prior assumption of histopathologic regional importance. All the patterns identified by HET match prior reporting on brain regions important for MSA-C and MSA-P. For further details refer to the citations given at the end of this page.

The code provided in this repo performs:

1. Model training (AutoGluon and 4 tree-based classifiers)
2. SHAP feature attribution
3. Weighting of regional feature construction
4. HET score computation

---

## 1. Concept

HET captures regional heterogeneity of MSA. The following are the patterns identified by volume (top row), fractional anisotropy (FA) (middle row), and mean diffusivity (MD) (bottom row) derived HET specific to MSA-C (left) and MSA-P (right)

<div align="center">
  <table>
    <tr>
      <th align="center"></th>
      <th align="center"><b>MSA-C</b></th>
      <th align="center"><b>MSA-P</b></th>
    </tr>
    <tr>
      <td align="center"><b>Volume</b></td>
      <td align="center">
        <img src="resources/gifs/msa-c-t1.gif" width="200" height="120" alt="GIF 1"/><br/>
        <sub><b>MSA-C Volume HET, t=0 → 1 year.</b></sub>
      </td>
      <td align="center">
        <img src="resources/gifs/msa-p-t1.gif" width="200" height="120" alt="GIF 2"/><br/>
        <sub><b>MSA-P Volume HET, t=0 → 1 year.</b></sub>
      </td>
    </tr>
    <tr>
      <td align="center"><b>FA</b></td>
      <td align="center">
        <img src="resources/gifs/msa-c-fa.gif" width="200" height="120" alt="GIF 3"/><br/>
        <sub><b>MSA-C FA HET, t=0 → 1 year.</b></sub>
      </td>
      <td align="center">
        <img src="resources/gifs/msa-p-fa.gif" width="200" height="120" alt="GIF 4"/><br/>
        <sub><b>MSA-P FA HET, t=0 → 1 year.</b></sub>
      </td>
    </tr>
    <tr>
      <td align="center"><b>MD</b></td>
      <td align="center">
        <img src="resources/gifs/msa-c-md.gif" width="200" height="120" alt="GIF 5"/><br/>
        <sub><b>MSA-C MD HET, t=0 → 1 year.</b></sub>
      </td>
      <td align="center">
        <img src="resources/gifs/msa-p-md.gif" width="200" height="120" alt="GIF 6"/><br/>
        <sub><b>MSA-P MD HET, t=0 → 1 year.</b></sub>
      </td>
    </tr>
  </table>
</div>

---

## 2. Getting started

#### Repo Structure

```
HET/
├── scripts/
│   ├── main.py                # Main pipeline controller
│   ├── models.py              # Model training with SGKF + AG/sklearn
│   └── feature_importance.py  # SHAP utilities and bootstrapping
├── example.csv                # Example input format
├── outputs/                   # Automatically generated results
├── resources/gifs
└── README.md
└── requirements.txt
```

#### First order of business: If you haven't already, install these packages using the following command:

```
pip install -r requirements.txt
```

---

## 3. Input Format

Your CSV file must contain:

- `ID` – subject identifier
- `visit` – numerical visit index
- `dx` – diagnostic label (`CON`, `MSA-C`, `MSA-P`, `PD`, etc.)
- feature columns – any numeric MRI variables

Example:
ID,visit,dx,feature_1,feature_2,...,feature_n
001,1,MSA-C,...
001,2,MSA-C,...

Feature names are automatically inferred by removing `ID`, `visit`, and `dx`.

---

## 4. Running the Pipeline

| Argument         | Description                            | Default  |
| ---------------- | -------------------------------------- | -------- |
| `--fresh_run`  | Train new models or load existing ones | True     |
| `--scaledata`  | Z-score features using controls        | True     |
| `--do_boot`    | Use bootstrap SHAP                     | True     |
| `--target`     | Target label column                    | dx       |
| `--model`      | Model identifier folder                | volume   |
| `--todays_run` | Custom run ID                          | YYYYMMDD |

* Model training (--fresh_run True)

```
python main.py --fresh_run True --scaledata True
```

* SHAP + HET (--fresh_run False)

```
python main.py --fresh_run False --do_boot True
```

---

## 4. Outputs

Model Outputs:

* Best classifier per seed
* Cross-validation metrics
* Best predictor folder

SHAP Outputs:

* shap_boot_mean_class_0.csv (or shap_values_class_0.csv)
* Class-specific SHAP CSVs
* Summary plots per class

HET Output:

* Final data with weighted features and HET scores: '*your_file_het.csv*'

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
Gebre, R.K., Raghavan, S., De Tora, M.E.J. et al. Precise disease heterogeneity and progression quantification in MSA and Parkinson’s disease using machine learning. Sci Rep 16, 10579 (2026). https://doi.org/10.1038/s41598-026-45949-5
```

---

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
