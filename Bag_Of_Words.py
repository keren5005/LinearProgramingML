import numpy as np

class BagOfWords:
    def __init__(self, documents):
        '''
        The constructor of the BagOfWords class takes a document as input. It then computes the frequency of each word in
        the document and selects the words with the highest frequency to form the dictionary.
        '''
        probs = {}
        total = 0
        for i in range(len(documents)):
            txt = documents[i]
            words = txt.split()
            for w1 in words:
                if any(c.isalpha() for c in w1):
                    w = w1.lower()
                    total = total + 1
                    if w not in probs:
                        probs[w] = 1
                    else:
                        probs[w] += 1
        freqs = [[w, probs[w] / total] for w in probs.keys()]
        freqs.sort(key=lambda x: x[1], reverse=True)
        total_prob = 0
        bag_of_words = set()
        for w, p in freqs:
            bag_of_words.add(w)
            total_prob = total_prob + p
            if total_prob > 0.95:
                break
        self.dictionary = {}
        k = 0
        for w in bag_of_words:
            self.dictionary[w] = k
            k += 1

    def encode(self, txt):
        '''
        The encode method takes a text document as input and encodes it as a vector
        of binary values using the dictionary generated in the constructor
        '''
        words = txt.split()
        x = np.zeros(len(self.dictionary))
        for w1 in words:
            w = w1.lower()
            if w in self.dictionary:
                pos = self.dictionary[w]
                x[pos] = 1
        return x

    def encode_documents(self, documents):
        '''
         takes a list of text documents as input and returns a numpy array where each row corresponds to a document
         and each column corresponds to a word in the dictionary. Each element of the array is either 0 or 1,
         indicating the presence or absence of a word in the corresponding document.
        '''
        out = []
        for i in range(len(documents)):
            txt = documents[i]
            out.append(self.encode(txt))
        x = np.array(out)
        return x
