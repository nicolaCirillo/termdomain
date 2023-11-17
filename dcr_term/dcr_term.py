import codecs
from itertools import chain
from collections import Counter
import pandas as pd
import numpy as np
import os
import requests
import tempfile
import zipfile
from itertools import dropwhile


from nltk.tree import Tree
from nltk import RegexpParser
from conllu import parse
from gensim.models import KeyedVectors

from dcr_term.tagger import Tagger

import os
ALACARTE = os.path.dirname(os.path.realpath(__file__)) + '/ALaCarte/alacarte.py'
FILEPATH = os.path.dirname(os.path.realpath(__file__)) + '/alacarte_files/'

def download(lang):
    if lang == "it":
        URL = "https://figshare.com/ndownloader/files/38520662"
        response = requests.get(URL)
        with tempfile.TemporaryFile() as tmp:
            tmp.write(response.content)
            with zipfile.ZipFile(tmp, 'r') as zip_ref:
                zip_ref.extractall(FILEPATH)
        

def fromtree(tree, extracted=list(), i=0):
    if i == len(tree):
        return
    node = tree[i]
    if type(node) == Tree:
        if node.label() == 'CAND':
            extracted.append(node.leaves())
        return fromtree(node, extracted), \
            fromtree(tree, extracted, i+1)
    else:
        return fromtree(tree, extracted, i+1)


class Extractor:
    def __init__(self, patterns):
        self.parser = RegexpParser(patterns)
    
    def extract(self, iterator):
        extr = list()
        for item in iterator:
            chunks = [item[i:i + 100] for i in range(0, len(item), 100)]
            for c in chunks:
                tree = self.parser.parse(c)
                fromtree(tree, extr)
        return extr


def read_conll(corpus):
    with codecs.open(corpus, "r","utf-8") as filein:
        sentences = parse(filein.read())
    iterator = list()
    for sent in sentences:
        item = [(token['form'].lower(), token['upostag']) for token in sent]
        iterator.append(item)
    return iterator


def extract_single_words(iterator):
    single_words = set(chain.from_iterable(iterator))
    single_words = ['_'.join(w) for w in single_words]
    return single_words

def load_path(lang):
    file = FILEPATH + lang + '/patterns.txt'
    with codecs.open(file, 'r', 'utf8') as filein:
        patterns = filein.read()
    return patterns
 
def extract_candidates(iterator, patterns):
    extractor = Extractor(patterns)
    candidates = extractor.extract(iterator)
    candidates = [' '.join(item[0] for item in c) for c in candidates]
    return candidates

# functions for contrastive weights
class Contrastive:
    def __init__(self, t_candidates, c_candidates):
        self.counter = Counter(t_candidates)
        self.c_counter = Counter(c_candidates)
        self.N_t = sum(Counter(t_candidates).values())
        self.N_c = sum(Counter(c_candidates).values())
        self.N = self.N_t + self.N_c

    def iwf(self, t: str):
        f_t = self.counter[t] + self.c_counter[t]
        try:
            return np.log2(self.N/f_t)
        except ZeroDivisionError:
            return self.N/1

    def t_weight(self, t):
        f_t = self.counter[t]
        return np.log2(f_t) * self.iwf(t)
    
    def ct_weight(self, ct):
        words = ct.split(' ')
        for w in words:
            if w.split('_')[1] == 'NOUN':
                h = w
                break
        return self.t_weight(h) * self.counter[ct]
    
    def tct_weight(self, t):
        if ' ' in t:
            return self.ct_weight(t)
        else:
            return self.t_weight(t)
    
    def tct_weight_2(self, t):
        f_t = self.counter[t]
        f_t2 = self.c_counter.get(t, 1)    
        return np.arctan(f_t/(f_t2/self.N_c))

    
    def get_fun(self, method):
        if method == 1:
            return self.tct_weight
        elif method == 2:
            return self.tct_weight_2

def contrastive(t_candidates, c_candidates, method=1):
    calc = Contrastive(t_candidates, c_candidates)
    scores = list()
    fun = calc.get_fun(method)
    for t in calc.counter:
        scores.append((t, fun(t)))
    return scores


def extract_contrastive(t_corpus, c_corpus, lang, method=1):
    patterns = load_path(lang)
    print("Extracting candidates from target corpus.", end =' ')
    t_iterator = read_conll(t_corpus)
    t_candidates = (extract_candidates(t_iterator, patterns))
    print("Done!\nExtracting candidates from contrastive corpus.", end =' ')
    c_iterator = read_conll(c_corpus)    
    c_candidates = (extract_candidates(c_iterator, patterns))
    print("Done!\ncomputing scores.", end=' ')
    scores = contrastive(t_candidates, c_candidates, method)
    print("Done!\nFinished")
    return sorted(scores, key=lambda x: x[1], reverse=True)

# end of function for contrastive weights

def gen_alacarte_files(iterator, targets, root):
    alc_corpus = root + 'corpus.txt'
    alc_targets = root + 'targets.txt'
    with codecs.open(alc_corpus, 'w', 'utf8') as fileout:
        fileout.writelines([' '.join(w[0] for w in sent) + '\n' 
                           for sent in iterator])
    with codecs.open(alc_targets, 'w', 'utf8') as fileout:
        targets = [c + '\n' for c in targets]
        fileout.writelines(sorted(targets))

def alacarte_vecs(lang, root):
    kwargs = {
        'alacarte': ALACARTE,
        'matrix': FILEPATH + lang + '/matrix_transform.bin',
        'vectors': FILEPATH + lang + '/source_vectors.txt',
        'corpus': root + 'corpus.txt',
        'targets_file': root + 'targets.txt',
        'dumproot': root + 'vectors'
    }
    comand = 'python {alacarte} -v -m {matrix} -s {vectors} -w 5 -c {corpus} -t {targets_file} {dumproot} --create-new'
    os.system(comand.format_map(kwargs))


def join_mwe(filename, dim=200):
    with codecs.open(filename, "r", "utf8") as filein:
        lines = filein.readlines()
    with codecs.open(filename, "w", "utf8") as fileout:
        for l in lines:
            l = l.split(" ")
            word, values = l[0:-dim], l[-dim:]
            if not all([float(v) == 0 for v in values]):
                fileout.write("_".join(word) + " " + " ".join(values))

def kcr(candidates, concepts, vectors, k=5):
    join_mwe(vectors)
    kv = KeyedVectors.load_word2vec_format(vectors, no_header=True)
    concepts = [c.replace(" ", "_") for c in concepts]
    scores = list()
    c_vecs = kv.vectors_for_all(concepts)
    c_vecs.fill_norms()
    for cand in candidates:
        try:
            v = kv.get_vector(cand.replace(" ", "_"))
        except KeyError:
            continue
        similarities = c_vecs.most_similar(v, topn=k)
        _, values = zip(*similarities)
        s = np.mean(values)
        scores.append((cand, s))
    return scores

def tag_corpus(corpus, tagger):
    tagged = '.'.join(corpus.split('.')[:-1])
    tagger.tag_doc(corpus, tagged)

    
def frequency_filter(candidates, min_freq):
    counter = Counter(candidates)
    for key, _ in dropwhile(lambda key_count: key_count[1] >= min_freq, 
                                counter.most_common()):
        del counter[key]
    return list(counter.keys())


def extract_terms(concepts: list, corpus: str, lang: str, fileroot,
                    k=5, gen_files=True):
    """Extract a list of terms from a specialized corpus.

    The extraction process is based on Domain Concept Relatedness (DCR). 
    It extract terms that are related to already-known concepts of a given 
    subbject field.
    
     Parameters
    ----------
    concepts: list of str
        A list of already-known terms of the subject field.
    corpus: str or path-like object
        The path to the specialized corpus (the corpus must be a .conll or
        a plain text file encoded in utf8).
    lang: str
        The language code.
    k: int, default=5
        k parameter for k-Nearest Neighbour.
    gen_files: bool, default=True
        if set to False, the function does not generate new word embeddings.


    """
    print("Extracting candidates.", end =' ')
    iterator = read_conll(corpus)
    patterns = load_path(lang)
    candidates = (extract_candidates(iterator, patterns))
    candidates = set(candidates)
    if gen_files:
        print("Done!\nGenerating embeddings.", end =' ')
        targets = set.union(set(concepts), candidates)
        gen_alacarte_files(iterator, targets, fileroot)
        alacarte_vecs(lang, fileroot)
    print("Done!\ncomputing scores.", end=' ')
    vectors = fileroot + 'vectors_alacarte.txt'
    scores = kcr(candidates, concepts, vectors, k=k)
    print("Done!\nFinished")
    return sorted(scores, key=lambda x: x[1], reverse=True)


def resuts2csv(results, path):
    """Saves the list of terms in .csv
    """
    df = pd.DataFrame(results)
    df.columns = ['term', 'score']
    df.to_csv(path, sep=';', index=False)