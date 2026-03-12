# Concept Drift resilience under integrated temporal variables study

# Overview
This RF model uses the kronodroid dataset with temporal variables embedded in to 1%< frequency privileges to train a malware detection model and analyze the effect of introducing temporal variables on model concept drift. 

Before use, the model requires libraries: os, numpy, pandas, and sklearn, with these libraries needing to be run in a virtual environment for sklearn. Create the virtual environment by navigating to the intended parent folder such as: "cd desktop/temporalResearchProj", and then run "python -m venv sklearn-env". Afterwards, run sklearn-env\Scripts\activate and "pip install numpy pandas matplotlib scikit-learn". 

Next, your IDE python compiler must use that of the virtual environment's. In VSC, this can be changed by typing ">python: select interpreter" in the top search bar. From here, ensure the RF python script and its accompanying kronodroid and privilege lifecycle datasets are all in the same folder, and run the kronoRF python script.

A, B, C models are produced, that test static variables exclusively, temporal variables exclusively, and both, respectively.

AUT: summary of overall performance


# Function

1. Data is loaded, noise columns and rows with invalid years are dropped

2. Temporal data is matched to privileges shared between both data sets. 

3. Temporal variables are created using temporal data:

| Feature | Description |
|---------|-------------|
| `perm_age_mean/max/min` | Average / max / min age of used permissions at app creation time |
| `perm_restricted_ratio` | Fraction of used permissions already restricted by the app year |
| `perm_deprecated_ratio` | Fraction of used permissions already deprecated by the app year |
| `perm_near_restrict_ratio` | Fraction with a known upcoming restriction within 2 years |
| `perm_restricted_count` | Count of already-restricted permissions |
| `perm_near_restrict_count` | Count of near-restriction permissions |
| `perm_deprecated_count` | Count of already-deprecated permissions |
| `perm_has_any_restricted` | Binary: app uses at least one restricted permission |
| `perm_has_any_near_restrict` | Binary: app uses at least one near-restriction permission |
| `perm_worst_age` | Age of the riskiest permission (closest to / most past restriction) |
| `perm_worst_perm_age_at_restrict` | Age of the riskiest permission at its restriction year |
| `perm_ttl_restrict_mean/min` | Mean / min years remaining until restriction (raw, before risk transform) |
| `perm_ttl_deprecate_mean/min` | Mean / min years remaining until deprecation (raw) |
| `perm_announced_restrict_ratio` | Fraction with a valid announced-but-not-yet-enacted restriction |
| `perm_ttl_announced_restrict_mean/min` | Mean / min years until the announced restriction |
| `perm_age_x_near_restrict` - `perm_worst_age × perm_near_restrict_ratio` | how long the riskiest permission has been in use as an upcoming restriction approaches, without dilution from the mean age.
| `perm_worst_age_x_risk_restrict` — `perm_worst_perm_age_at_restrict × perm_risk_worst_restrict`| joint signal from the age and risk score of the single riskiest permission.

*need to determine if setting temporal feature values to zero negatively impacts C model through noise / ambiguity*

4. Privilege risk scored based on time to / past restriction and deprecation, with restriction being higher risk

5. Binary permission flag, temporal, year columns whitelisted

6. Model trained on years 2008-2014, tested on 2014-2017. 2018 excluded due to unreasonably high malware:benign ratio (93% malware)

7. A, B, C models are evaluated on Macro F1, Balanced Accuracy, Weighted F1 metrics. 

Example table:
```
======================================================================
Metric: [metric]                (AUT over k=1–3, excluding 2018)
  Model                          k=1    k=2    k=3    k=4    AUT
  ─────────────────────────────────────────────────────────────
  [A] Static flags only          0.XXX  0.XXX  0.XXX  0.XXX  0.XXXX
  [B] Temporal only              0.XXX  0.XXX  0.XXX  0.XXX  0.XXXX
  [C] Count-free full model      0.XXX  0.XXX  0.XXX  0.XXX  0.XXXX
=======================================================================
```
| Metric | Why |
|--------|-----|
| **Macro F1** | Primary AUT metric; treats malware and benign classes equally regardless of size |
| **Balanced Accuracy** | Accounts for class imbalance without averaging F1 across classes |
| **Weighted F1** | Reflects overall dataset-weighted performance |
| **Accuracy / Precision / Recall** | Printed per fold for reference; not used in AUT |

# Limitations, Future work:

1. Based on single fixed training window

2. Dataset becomes more malware concentrated as it progresses. By 2018, only 30 samples are benign. 93% are malware.

3. No real device data yet

4. Only RF implemented so far

5. Make this file itself clearer and more obvious every step on the way

6. Stratify kronoRF- separate files/folder for models, data, make more MODULAR
