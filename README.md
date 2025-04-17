# ❤️ Leveraging the TOTEM algorithm for prediction of a moving base of a ship on oceanic waves. 
  - This is used as an ablation study to the Koopman based model to compare which methods better capture quasi-periodic motion for multi-step prediction. 

# TOTEM: TOkenized Time series EMbeddings for General Time Series Analysis
TOTEM explores time series unification through discrete tokens (not patches!!). Its simple VQVAE backbone learns a self-supervised, discrete, codebook in either a generalist (multiple domains) or specialist (1 domain) manner.
TOTEM's codebook can then be tested on in domain or zero shot data with many 🔥 time series tasks.

For a high level overview see the [video recap](https://www.youtube.com/watch?v=OqrCpdb6MJk).
Check out the [paper](https://arxiv.org/pdf/2402.16412.pdf) for more details!

## Get Started with TOTEM 💪

### 1. Setup your environment 🤓
```
pip install -r requirements.txt
```

### 2. Run TOTEM 🚀

```
# Forecasting Specialist
forecasting/scripts/base.sh

# Process Zero Shot Data
process_zero_shot_data/scripts/base.sh
```

## Cite If You ❤️ TOTEM

```
@article{
talukder2024totem,
title={{TOTEM}: {TO}kenized Time Series {EM}beddings for General Time Series Analysis},
author={Sabera J Talukder and Yisong Yue and Georgia Gkioxari},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=QlTLkH6xRC},
note={}
}
