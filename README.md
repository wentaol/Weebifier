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

## Usage and samples
```python
from weebifier import Weebifier
w = Weebifier()
w.weebify("hello world")
w.weebify("some fake words:" )
w.weebify("nelinetrolls hevephiny lantifices")
w.weebify("and some harry potter spells:")
w.weebify("expelliarmus wingardium leviosa")
```

Output: 

```
ヘロー ワールド
ソム フェイク ワーズ :
ネライントロールズ ヘビフィニー ランタファイシズ
アンド ソム ハリー ポター スペルズ :
エクスペリャルムス ウィンガーディアム レビオーサ
```