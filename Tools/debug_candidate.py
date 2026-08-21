from core.candidates import generate_and_score
import json

def show(text):
    res = generate_and_score(text)
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    show('Take care of the thing from before')
