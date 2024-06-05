Codes for *Harnessing Physicians’ Propensities within EHR Data to Enhance the Objectivity of AI-Driven Clinical Decision-Making*

Codes can be implemented on the MIMIC-IV database by following:

1. Download MIMIC-IV (ver 2.2) from https://physionet.org/content/mimiciv/2.2/.
2. Run `MIMIC_data_extraction.ipynb` to get the used MIMIC-IV dataset.
3. Run `grid search.ipynb` to determine the hyperparameters using baseline.
4. Run `training.ipynb` to train the baseline and propensity-harnessed models.
5. Run `evaluation.ipynb` to get the evaluation results and figures presented in paper.

Codes can also be implemented on private datasets by structuring data into two Dataframes: `df_feature` and `df_label`, with their first index level being `provider_id`.
