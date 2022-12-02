import codecs
import warnings

from spacy_conll import init_parser

class Tagger:
    def __init__(self, lang: str):
        self.nlp = init_parser(
            lang, "udpipe",
            parser_opts={"use_gpu": True, "verbose": True},
            include_headers=True)

    def tag_string(self, string: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokens = self.nlp(string)
        return ' '.join(t.text + '_'+ t.pos_ for t in tokens)

    def tag_doc(self, filein, fileout):
        with codecs.open(filein, 'r', 'utf8') as filein:
            raw = filein.read()
            sents = raw.split('\n')
            text = ''
            for string in sents:
                if string:
                    if string[0] == '#':
                        text += string + '\n'
                    else:
                        conll = self.nlp(string)._.conll_str
                        text += conll + '\n'
        with codecs.open(fileout, 'w', 'utf8') as fileout:
            fileout.write(text)
