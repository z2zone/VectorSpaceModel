import numpy as np
import re
import math
from gensim.models import KeyedVectors
from sklearn.utils.extmath import randomized_svd
from collections import Counter


def top_k_unigrams(tweets, stop_words, k):
    pattern = re.compile(r'^[a-z#]\w*$')
    counter = Counter()
    #parse
    for tweet in tweets:
        words = tweet.split()
        for word in words:
            word = word.lower()
            if word in stop_words or not pattern.match(word):
                continue
            counter[word] += 1
    if k == -1:
        return counter
    top_words = counter.most_common(k)
    return dict(top_words)

def context_word_frequencies(tweets, stop_words, context_size, frequent_unigrams):
    pattern = re.compile(r'^[a-z#]\w*$')
    counter = Counter()

    for tweet in tweets:
        words = tweet.split()
        #need index to locate the word
        for i, word in enumerate(words):
            word = word.lower()
            #stopwords need to be removed
            if word in stop_words or not pattern.match(word):
                continue
            if word not in frequent_unigrams:
                continue
            start_index = max(0, i - context_size)
            end_index = min(len(words) - 1, i + context_size)
            for j in range(start_index, end_index + 1):
                # Skipp
                if j == i or words[j] in stop_words:
                    continue
                context_word = words[j].lower()
                # Skip words that is unecessary
                if not pattern.match(context_word) or context_word not in frequent_unigrams:
                    continue
                word_pair = (word, context_word)
                counter[word_pair] += 1

    return Counter(dict(counter))

def pmi(word1, word2, unigram_counter, context_counter):
    bigram_freq = context_counter.get((word1, word2), 0)
    unigram1 = unigram_counter[word1]
    unigram2 = unigram_counter[word2]
    total = sum(unigram_counter.values())

    if bigram_freq == 0:
        return 0

    frequency = (unigram1 * unigram2) / total
    if frequency == 0:
        return 0

    pmi_score = math.log2((bigram_freq + 1) / frequency)
    return pmi_score

def build_word_vector(word1, frequent_unigrams, unigram_counter, context_counter):
    word_vector = {}
    for dimension in frequent_unigrams:
        if dimension != word1:
            pmi_score = pmi(word1, dimension, unigram_counter, context_counter)
            word_vector[dimension] = pmi_score if pmi_score else 0
    return word_vector

def get_top_k_dimensions(word1_vector, k):
    sort = sorted(word1_vector.items(), key=lambda x: x[1], reverse=True)
    top_k_dimensions = {}
    for i in range(k):
        if i >= len(sort):
            break
        #switch
        top_k_dimensions[sort[i][0]] = sort[i][1]
    return top_k_dimensions

def get_cosine_similarity(word1_vector, word2_vector):
    dot_product = 0
    magnitude_word1 = 0
    magnitude_word2 = 0
    
    #from wikipedia's equation
    for word in word1_vector:
        if word in word2_vector:
            dot_product += word1_vector[word] * word2_vector[word]
        magnitude_word1 += word1_vector[word] ** 2
    for word in word2_vector:
        magnitude_word2 += word2_vector[word] ** 2

    # doesn't make sense to have 0 vecotr?
    if magnitude_word1 == 0 or magnitude_word2 == 0:
        return 0

    cosine_similarity = dot_product / (math.sqrt(magnitude_word1) * math.sqrt(magnitude_word2))

    return cosine_similarity

def get_most_similar(word2vec, word, k):
    if word not in word2vec:
        return []
    else:
        return word2vec.similar_by_word(word, topn=k)

def word_analogy(word2vec, word1, word2, word3):
    if word1 not in word2vec or word2 not in word2vec or word3 not in word2vec:
        return None
    analogy_vector = word2vec[word2] - word2vec[word1] + word2vec[word3]

    # answer tuple
    answ, score = word2vec.most_similar(positive=[analogy_vector], topn=1)[0]

    return (answ, score)


def create_tfidf_matrix(documents, stopwords):
    preprocessed_docs = []
    for doc in documents:
        preprocessed_doc = [word.lower() for word in doc if word.lower() not in stopwords and word.isalnum()]
        preprocessed_docs.append(preprocessed_doc)
    
    vocab = sorted(set([word for doc in preprocessed_docs for word in doc]))
    num_docs = len(preprocessed_docs)
    num_words = len(vocab)
    
    tf_matrix = np.zeros((num_docs, num_words))
    for i, doc in enumerate(preprocessed_docs):
        word_freq = Counter(doc)
        for j, word in enumerate(vocab):
            tf_matrix[i][j] = word_freq[word]
    
    doc_freq = np.sum(tf_matrix > 0, axis=0)
    idf_matrix = np.log10(num_docs / (doc_freq + 1))  # Smoothing IDF
    
    tfidf_matrix = tf_matrix * idf_matrix
    
    return tfidf_matrix, vocab

def get_idf_values(documents, stopwords):
    # This part is ungraded, however, to test your code, you'll need to implement this function
    # If you have implemented create_tfidf_matrix, this implementation should be straightforward
    # FILL IN CODE
    pass

def calculate_sparsity(tfidf_matrix):
    total_cells = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
    zero_cells = np.sum(tfidf_matrix == 0)
    sparsity = zero_cells / total_cells
    return sparsity

def extract_salient_words(VT, vocabulary, k):
    salient_words = {}
    for i in range(VT.shape[0]):
        dimension_words = {}
        dimension_values = VT[i]
        sorted_indices = np.argsort(dimension_values)[::-1][:k]
        for j in sorted_indices:
            word = vocabulary[j]
            score = dimension_values[j]
            dimension_words[word] = score
        salient_words[i] = dimension_words
    return salient_words

def get_similar_documents(U, Sigma, VT, doc_index, k):
    # FILL IN CODE
    pass

def document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, k):
    # FILL IN CODE
    pass

if __name__ == '__main__':
    
    tweets = []
    with open('/Users/brandon/Desktop/etc/template/data/covid-tweets-2020-08-10-2020-08-21.tokenized.txt') as f:
        tweets = [line.strip() for line in f.readlines()]

    stop_words = []
    with open('/Users/brandon/Desktop/etc/template/data/stop_words.txt') as f:
        stop_words = [line.strip() for line in f.readlines()]


    # """Building Vector Space model using PMI"""

    # print(top_k_unigrams(tweets, stop_words, 10))
    # # {'covid': 71281, 'pandemic': 50353, 'covid-19': 33591, 'people': 31850, 'n’t': 31053, 'like': 20837, 'mask': 20107, 'get': 19982, 'coronavirus': 19949, 'trump': 19223}
    # frequent_unigrams = top_k_unigrams(tweets, stop_words, 1000)
    # unigram_counter = top_k_unigrams(tweets, stop_words, -1)
    
    # ### THIS PART IS JUST TO PROVIDE A REFERENCE OUTPUT
    # sample_output = context_word_frequencies(tweets, stop_words, 2, frequent_unigrams)
    # print(sample_output.most_common(10))
    # """
    # [(('the', 'pandemic'), 19811),
    # (('a', 'pandemic'), 16615),
    # (('a', 'mask'), 14353),
    # (('a', 'wear'), 11017),
    # (('wear', 'mask'), 10628),
    # (('mask', 'wear'), 10628),
    # (('do', 'n’t'), 10237),
    # (('during', 'pandemic'), 8127),
    # (('the', 'covid'), 7630),
    # (('to', 'go'), 7527)]
    # """
    # ### END OF REFERENCE OUTPUT
    
    # context_counter = context_word_frequencies(tweets, stop_words, 3, frequent_unigrams)

    # word_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    # print(get_top_k_dimensions(word_vector, 10))
    # # {'put': 6.301874856316369, 'patient': 6.222687002250096, 'tried': 6.158108051673095, 'wearing': 5.2564459708663875, 'needed': 5.247669358807432, 'spent': 5.230966480014661, 'enjoy': 5.177980198384708, 'weeks': 5.124941187737894, 'avoid': 5.107686157639801, 'governors': 5.103879572210065}

    # word_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    # print(get_top_k_dimensions(word_vector, 10))
    # # {'wear': 7.278203356425305, 'wearing': 6.760722107602916, 'mandate': 6.505074539073231, 'wash': 5.620700962265705, 'n95': 5.600353617179614, 'distance': 5.599542578641884, 'face': 5.335677912801717, 'anti': 4.9734651502193366, 'damn': 4.970725788331299, 'outside': 4.4802694058646}

    # word_vector = build_word_vector('distancing', frequent_unigrams, unigram_counter, context_counter)
    # print(get_top_k_dimensions(word_vector, 10))
    # # {'social': 8.637723567642842, 'guidelines': 6.244375965192868, 'masks': 6.055876420939214, 'rules': 5.786665161219354, 'measures': 5.528168931193456, 'wearing': 5.347796214635814, 'required': 4.896659865603407, 'hand': 4.813598338358183, 'following': 4.633301876715461, 'lack': 4.531964710683777}

    # word_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    # print(get_top_k_dimensions(word_vector, 10))
    # # {'donald': 7.363071158640809, 'administration': 6.160023745590209, 'president': 5.353905139926054, 'blame': 4.838868198365827, 'fault': 4.833928177006809, 'calls': 4.685281547339574, 'gop': 4.603457978983295, 'failed': 4.532989597142956, 'orders': 4.464073158650432, 'campaign': 4.3804665561680824}

    # word_vector = build_word_vector('pandemic', frequent_unigrams, unigram_counter, context_counter)
    # print(get_top_k_dimensions(word_vector, 10))
    # # {'global': 5.601489175269805, 'middle': 5.565259949326977, 'amid': 5.241312533124981, 'handling': 4.609483077248557, 'ended': 4.58867551721951, 'deadly': 4.371399989758025, 'response': 4.138827482426898, 'beginning': 4.116495953781218, 'pre': 4.043655804452211, 'survive': 3.8777495603541254}

    # word1_vector = build_word_vector('ventilator', frequent_unigrams, unigram_counter, context_counter)
    # word2_vector = build_word_vector('covid-19', frequent_unigrams, unigram_counter, context_counter)
    # print(get_cosine_similarity(word1_vector, word2_vector))
    # # 0.2341567704935342

    # word2_vector = build_word_vector('mask', frequent_unigrams, unigram_counter, context_counter)
    # print(get_cosine_similarity(word1_vector, word2_vector))
    # # 0.05127326904936171

    # word1_vector = build_word_vector('president', frequent_unigrams, unigram_counter, context_counter)
    # word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    # print(get_cosine_similarity(word1_vector, word2_vector))
    # # 0.7052644362543867

    # word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    # print(get_cosine_similarity(word1_vector, word2_vector))
    # # 0.6144272810573133

    # word1_vector = build_word_vector('trudeau', frequent_unigrams, unigram_counter, context_counter)
    # word2_vector = build_word_vector('trump', frequent_unigrams, unigram_counter, context_counter)
    # print(get_cosine_similarity(word1_vector, word2_vector))
    # # 0.37083874436657593

    # word2_vector = build_word_vector('biden', frequent_unigrams, unigram_counter, context_counter)
    # print(get_cosine_similarity(word1_vector, word2_vector))
    # # 0.34568665086152817


    # """Exploring Word2Vec"""

    # EMBEDDING_FILE = '/Users/brandon/Desktop/etc/template/data/GoogleNews-vectors-negative300.bin.gz'
    # word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

    # similar_words =  get_most_similar(word2vec, 'ventilator', 3)
    # print(similar_words)
    # # [('respirator', 0.7864563465118408), ('mechanical_ventilator', 0.7063839435577393), ('intensive_care', 0.6809945702552795)]

    # # Word analogy - Tokyo is to Japan as Paris is to what?
    # print(word_analogy(word2vec, 'Tokyo', 'Japan', 'Paris'))
    # # ('France', 0.7889978885650635)


    # """Latent Semantic Analysis"""
    import ssl
    import nltk
    nltk.download('brown')
    from nltk.corpus import brown
    documents = [brown.words(fileid) for fileid in brown.fileids()]

    # Exploring the corpus
    print("The news section of the Brown corpus contains {} documents.".format(len(documents)))
    for i in range(3):
        document = documents[i]
        print("Document {} has {} words: {}".format(i, len(document), document))
    # The news section of the Brown corpus contains 500 documents.
    # Document 0 has 2242 words: ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
    # Document 1 has 2277 words: ['Austin', ',', 'Texas', '--', 'Committee', 'approval', ...]
    # Document 2 has 2275 words: ['Several', 'defendants', 'in', 'the', 'Summerdale', ...]

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')

    # This will take a few minutes to run
    tfidf_matrix, vocabulary = create_tfidf_matrix(documents, stopwords_list)
    idf_values = get_idf_values(documents, stopwords_list)

    print(tfidf_matrix.shape)
    # (500, 40881)

    print(tfidf_matrix[np.nonzero(tfidf_matrix)][:5])
    # [5.96857651 2.1079054  3.         2.07572071 2.69897   ]

    print(vocabulary[2000:2010])
    # ['amoral', 'amorality', 'amorist', 'amorous', 'amorphous', 'amorphously', 'amortization', 'amortize', 'amory', 'amos']

    print(calculate_sparsity(tfidf_matrix))
    # 0.9845266994447298

    """SVD"""
    U, Sigma, VT = randomized_svd(tfidf_matrix, n_components=10, n_iter=100, random_state=42)

    salient_words = extract_salient_words(VT, vocabulary, 10)
    print(salient_words[1])
    # ['anode', 'space', 'theorem', 'v', 'q', 'c', 'p', 'operator', 'polynomial', 'af']

    print("We will fetch documents similar to document {} - {}...".format(3, ' '.join(documents[3][:50])))
    # We will fetch documents similar to document 3 - 
    # Oslo The most positive element to emerge from the Oslo meeting of North Atlantic Treaty Organization Foreign Ministers has been the freer , 
    # franker , and wider discussions , animated by much better mutual understanding than in past meetings . This has been a working session of an organization that...

    similar_doc_indices = get_similar_documents(U, Sigma, VT, 3, 5)
    for i in range(2):
        print("Document {} is similar to document 3 - {}...".format(similar_doc_indices[i], ' '.join(documents[similar_doc_indices[i]][:50])))
    # Document 61 is similar to document 3 - 
    # For a neutral Germany Soviets said to fear resurgence of German militarism to the editor of the New York Times : 
    # For the first time in history the entire world is dominated by two large , powerful nations armed with murderous nuclear weapons that make conventional warfare of the past...
    # Document 6 is similar to document 3 - 
    # Resentment welled up yesterday among Democratic district leaders and some county leaders at reports that Mayor Wagner had decided to seek a third term with Paul R. Screvane and Abraham D. Beame as running mates . 
    # At the same time reaction among anti-organization Democratic leaders and in the Liberal party... 
    
    query = ['Krim', 'attended', 'the', 'University', 'of', 'North', 'Carolina', 'to', 'follow', 'Thomas', 'Wolfe']
    print("We will fetch documents relevant to query - {}".format(' '.join(query)))
    relevant_doc_indices = document_retrieval(vocabulary, idf_values, U, Sigma, VT, query, 5)
    for i in range(2):
        print("Document {} is relevant to query - {}...".format(relevant_doc_indices[i], ' '.join(documents[relevant_doc_indices[i]][:50])))
    # Document 90 is relevant to query - 
    # One hundred years ago there existed in England the Association for the Promotion of the Unity of Christendom . 
    # Representing as it did the efforts of only unauthorized individuals of the Roman and Anglican Churches , and urging a communion of prayer unacceptable to Rome , this association produced little...
    # Document 101 is relevant to query - To what extent and in what ways did Christianity affect the United States of America in the nineteenth century ? ? 
    # How far and in what fashion did it modify the new nation which was emerging in the midst of the forces shaping the revolutionary age ? ? To what...
