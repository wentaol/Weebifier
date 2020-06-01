# Weebifier
Converts English (and not so English) words to Katana with a seq2seq model.
Optionally use override network predictions with a dictionary of word mappings.

Based on [this]https://github.com/keon/seq2seq
Trained with data from [here]https://github.com/jamesohortle/loanwords_gairaigo

## Requirements
- numpy
- torch

For preprocessing data:
- sqlite3
- pandas

For training:
- torchtext
