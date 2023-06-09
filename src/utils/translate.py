import langid
import sys

def identify_languages(text):
    lang, prob = langid.classify(text)
    return {
        'language': lang,
        'probability': prob
    }


if __name__ == '__main__':
    text = sys.argv[1]
    print(identify_languages(text))
        