
import sys
from weebifier import Weebifier

if __name__ == "__main__":
    w = Weebifier()
    test_cases = ["some simple text to convert",
                  "rudimentary... handling of punctuation!",
                  "neo armstrong cyclone jet armstrong 砲",
                  "nelinetrolls hevephiny lantifices culatent",
                  "riddikulus sectumsempra avada kedavra",
                  "alohomora expelliarmus wingardium leviosa",
                  "expecto patronum",
                ]
    for sentence in test_cases:
        print("SENTENCE:",sentence)
        print("NET ONLY:", w.weebify(sentence, use_db=False))
        print("DB ONLY :", w.weebify(sentence, use_net=False))
        print("DB + NET:", w.weebify(sentence))
        print("---------")
    


