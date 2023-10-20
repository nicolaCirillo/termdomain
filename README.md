CONTENTS OF THIS FILE
---------------------

*   Introduction
*   Setup
*   Getting started

INTRODUCTION
------------

A semi-supervised domain-aware term extractor. If you use this software in your research, please cite:
```
@inproceedings{cirillo2023isolating,
  author = {N. Cirillo},
  booktitle = {Proceedings of the 2nd International Conference on Multilingual Digital Terminology Today (MDTT 2023)},
  publisher = {CEUR Workshop Proceedings},
  title = {Isolating Terminology Layers in Complex Linguistic Environments: a Study About Waste Management (Short Paper)},
  volume = {3427},
  url = {https://ceur-ws.org/Vol-3427/short3.pdf},
  year = {2023},
}
```

SETUP
-----
```
pip install git+https://github.com/nicolaCirillo/termdomain
```
### Download the language module
```
python -c "from dcr_term import dcr_term; dcr_term.download('it')"
```
GETTING STARTED
---------------
```
from dcr_term import extract_terms, resuts2csv

key_concepts = ["acque reflue", "bilancio di sostenibilità", "biodegradabile",
    "centro di raccolta", "compostaggio", "discarica", "effetto serra",
    "emissione", "frazione multimateriale", "imballaggio", "termovalorizzatore"
    ]
corpus = 'files/eval_data/corpus.conllu'
lang = 'it'
fileroot = 'files/sample_'
k = 5
terms = extract_terms(key_concepts, corpus, lang, fileroot, k=k)
resuts2csv(terms, 'files/dcr.csv')
```
