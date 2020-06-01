# Weebifier
Converts English (and not so English) words to Katana with a seq2seq model.
Optionally override network predictions with a dictionary of word mappings provided as csv.

Based on [this](https://github.com/keon/seq2seq), trained with data from [here](https://github.com/jamesohortle/loanwords_gairaigo)

## Requirements
- numpy
- torch

For preprocessing data:
- sqlite3
- pandas

For training:
- torchtext
