import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import nltk



import matplotlib.pyplot as plt
import seaborn as sns
#from IPython.display import Image




st = stopwords.words('english')
stemmer = PorterStemmer()

def loadDataAsDataFrame(f_path):
    df = pd.read_csv(f_path)
    return df

word_clusters = {}

def loadwordclusters():
    infile = open('./50mpaths2.txt')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters

def getclusterfeatures(sent):
    sent = sent.lower()
    terms = nltk.word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)

def preprocess_text(raw_text):
    #stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))

def grid_search_hyperparam_space(params, pipeline, folds, training_texts, training_classes):#folds, x_train, y_train, x_validation, y_validation):
        grid_search = GridSearchCV(estimator=pipeline, param_grid=params, refit=True, cv=folds, return_train_score=False, scoring='f1_micro',n_jobs=-1)
        grid_search.fit(training_texts, training_classes)
        return grid_search


if __name__ == '__main__':
    # Load the data
    f_path = './pdfalls.csv'
    data = loadDataAsDataFrame(f_path)
    texts = data['fall_description']
    classes = data['fall_class']
    classes = classes.replace("BoS", "Other")   #binary classification
    age = data['age']
    location = data['fall_location']


    # ADD GENDER AS A FEATURE
    gender = []
    for row in data['female']:
        if row == 'Female':
            gender.append(str(0))
        else:
            gender.append(str(1))

    #gender = pd.Series(gender)



    # SPLIT THE DATA
    training_set_size = int(0.8 * len(data))
    training_data = data[:training_set_size]
    training_texts = texts[:training_set_size]
    training_classes = classes[:training_set_size]
    training_age = age[:training_set_size]
    training_gender = gender[:training_set_size]
    training_location = location[:training_set_size]

    test_data = data[training_set_size:]
    test_texts = texts[training_set_size:]
    test_classes = classes[training_set_size:]
    test_age = age[training_set_size:]
    test_gender = gender[training_set_size:]
    test_location = location[training_set_size:]

    # PREPROCESS THE DATA
    training_texts_preprocessed = []
    test_texts_preprocessed = []
    test_clusters = []
    training_clusters = []
    training_length = []
    test_length = []
    training_age_preprocessed = []
    test_age_preprocessed = []

    word_clusters = loadwordclusters()

    for tr in training_texts:
        # you can do more with the training text here and generate more features...
        training_texts_preprocessed.append(preprocess_text(tr))
        training_clusters.append(getclusterfeatures(tr))
        training_length.append(str(len(tr)))
    for tt in test_texts:
        test_texts_preprocessed.append(preprocess_text(tt))
        test_clusters.append(getclusterfeatures(tt))
        test_length.append(str(len(tt)))
    for tr in training_age:
        training_age_preprocessed.append(str(tr))
    for tt in test_age:
        test_age_preprocessed.append(str(tt))



    # 10-FOLD CROSS VALIDATION
    skf = StratifiedKFold(n_splits=10)

    # FEATURE: NGRAMS
    print("-------NGRAMS-------")
    skf.get_n_splits(training_texts_preprocessed, training_classes)
    for train_index, test_index in skf.split(training_texts_preprocessed, training_classes):
        training_texts_preprocessed_train = map(training_texts_preprocessed.__getitem__, train_index)
        training_texts_preprocessed_dev = map(training_texts_preprocessed.__getitem__, test_index)



        ttp_train, ttp_test = training_classes[train_index], training_classes[test_index]

        # VECTORIZER
        vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000)
        training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed_train).toarray()
        test_data_vectors = vectorizer.transform(training_texts_preprocessed_dev).toarray()

        print(".......")

        # NAIVE BAYES CLASSIFIER
        gnb = GaussianNB()
        gnb_classifier = gnb.fit(training_data_vectors, ttp_train)
        gnb_predictions = gnb.predict(test_data_vectors)
        print("NAIVE BAYES f1-micro:", f1_score(ttp_test, gnb_predictions, average = 'micro'))
        print("NAIVE BAYES f1-macro:", f1_score(ttp_test, gnb_predictions, average = 'macro'))
        print("NAIVE BAYES accuracy:", accuracy_score(ttp_test, gnb_predictions))



        # UNOPTIMIZED SVM CLASSIFIER
        svm_unop = svm.SVC(C=1, cache_size=200,
                           coef0=0.0, degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=True,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)
        svm_unop_classifier = svm_unop.fit(training_data_vectors, ttp_train)
        svm_unop_predictions = svm_unop.predict(test_data_vectors)
        print("UNOPTIMIZED SVM f1-micro:", f1_score(ttp_test, svm_unop_predictions, average = 'micro'))
        print("UNOPTIMIZED SVM f1-macro:", f1_score(ttp_test, svm_unop_predictions, average = 'macro'))
        print("UNOPTIMIZED SVM accuracy:", accuracy_score(ttp_test, svm_unop_predictions))



        # RANDOM FOREST CLASSIFIER
        rf = RandomForestClassifier(n_estimators=20, random_state=1)
        rf_classifier = rf.fit(training_data_vectors, ttp_train)
        rf_predictions = rf.predict(test_data_vectors)
        print("RANDOM FOREST f1-micro:", f1_score(ttp_test, rf_predictions, average = 'micro'))
        print("RANDOM FOREST f1-macro:", f1_score(ttp_test, rf_predictions, average = 'macro'))
        print("RANDOM FOREST accuracy:", accuracy_score(ttp_test, rf_predictions))


        # K NEAREST NEIGHBORS CLASSIFIER
        grid_params = {
            'knn__n_neighbors': [1, 2, 3, 4, 5],
        }
        knn = KNeighborsClassifier()
        folds = 10
        pipeline = Pipeline(steps=[('vec', vectorizer), ('knn', knn)])
        grid = grid_search_hyperparam_space(grid_params, pipeline, folds, training_texts_preprocessed, training_classes)
        n_neighbors_ = grid.best_params_['knn__n_neighbors']

        knn_classifier = knn.fit(training_data_vectors, ttp_train)
        knn_predictions = knn.predict(test_data_vectors)
        print("K NEAREST NEIGHBORS f1-micro:", f1_score(ttp_test, knn_predictions, average = 'micro'))
        print("K NEAREST NEIGHBORS f1-macro:", f1_score(ttp_test, knn_predictions, average = 'macro'))
        print("K NEAREST NEIGHBORS accuracy:", accuracy_score(ttp_test, knn_predictions))


        # LOGISTIC REGRESSION CLASSIFIER
        lr = LogisticRegression(random_state=0)
        lr_classifier = lr.fit(training_data_vectors, ttp_train)
        lr_predictions = lr.predict(test_data_vectors)
        print("LOGISTIC REGRESSION f1-micro:", f1_score(ttp_test, lr_predictions, average = 'micro'))
        print("LOGISTIC REGRESSION f1-macro:", f1_score(ttp_test, lr_predictions, average = 'macro'))
        print("LOGISTIC REGRESSION accuracy:", accuracy_score(ttp_test, lr_predictions))


        # NEURAL NETWORK
        nn = MLPClassifier(random_state=1, max_iter=300)
        nn_classifier = nn.fit(training_data_vectors, ttp_train)
        nn_predictions = nn.predict(test_data_vectors)
        print("NEURAL NETWORK f1-micro:", f1_score(ttp_test, nn_predictions, average = 'micro'))
        print("NEURAL NETWORK f1-macro:", f1_score(ttp_test, nn_predictions, average = 'macro'))
        print("NEURAL NETWORK accuracy:", accuracy_score(ttp_test, nn_predictions))


        # VOTING CLASSIFIER
        ens = VotingClassifier(estimators=[('rf', rf), ('gnb', gnb), ('svm_unop', svm_unop)], voting='hard')
        ens_classifier = ens.fit(training_data_vectors, ttp_train)
        ens_predictions = ens.predict(test_data_vectors)
        print("VOTING ENSEMBLE f1-micro:", f1_score(ttp_test, ens_predictions, average = 'micro'))
        print("VOTING ENSEMBLE f1-macro:", f1_score(ttp_test, ens_predictions, average = 'macro'))
        print("VOTING ENSEMBLE accuracy:", accuracy_score(ttp_test, ens_predictions))


    # FEATURE: GENDER
    print("-------GENDER-------")
    skf.get_n_splits(training_gender, training_classes)
    for train_index, test_index in skf.split(training_gender, training_classes):
        training_gender_train = map(training_gender.__getitem__, train_index)
        training_gender_dev = map(training_gender.__getitem__, test_index)

        ttp_train, ttp_test = training_classes[train_index], training_classes[test_index]


        vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000, stop_words=None, token_pattern=r"(?u)\b\w+\b")

        training_gender_vectors = vectorizer.fit_transform(training_gender_train).toarray()
        test_gender_vectors = vectorizer.transform(training_gender_dev).toarray()

        print(".......")
        # NAIVE BAYES CLASSIFIER
        gnb_gender_classifier = gnb.fit(training_gender_vectors, ttp_train)
        gnb_gender_predictions = gnb.predict(test_gender_vectors)
        print("NAIVE BAYES f1-micro:", f1_score(ttp_test, gnb_gender_predictions, average='micro'))
        print("NAIVE BAYES f1-macro:", f1_score(ttp_test, gnb_gender_predictions, average='macro'))
        print("NAIVE BAYES accuracy:", accuracy_score(ttp_test, gnb_gender_predictions))

        # UNOPTIMIZED SVM CLASSIFIER
        svm_unop_gender_classifier = svm_unop.fit(training_gender_vectors, ttp_train)
        svm_unop_gender_predictions = svm_unop.predict(test_gender_vectors)
        print("UNOPTIMIZED SVM f1-micro:", f1_score(ttp_test, svm_unop_gender_predictions, average='micro'))
        print("UNOPTIMIZED SVM f1-macro:", f1_score(ttp_test, svm_unop_gender_predictions, average='macro'))
        print("UNOPTIMIZED SVM accuracy:", accuracy_score(ttp_test, svm_unop_gender_predictions))

        # RANDOM FOREST CLASSIFIER
        rf_gender_classifier = rf.fit(training_gender_vectors, ttp_train)
        rf_gender_predictions = rf.predict(test_gender_vectors)
        print("RANDOM FOREST f1-micro:", f1_score(ttp_test, rf_gender_predictions, average='micro'))
        print("RANDOM FOREST f1-macro:", f1_score(ttp_test, rf_gender_predictions, average='macro'))
        print("RANDOM FOREST accuracy:", accuracy_score(ttp_test, rf_gender_predictions))

        # K NEAREST NEIGHBORS CLASSIFIER
        grid_params = {
            'knn__n_neighbors': [1, 2, 3, 4, 5],
        }
        folds = 10
        pipeline = Pipeline(steps=[('vec', vectorizer), ('knn', knn)])
        grid = grid_search_hyperparam_space(grid_params, pipeline, folds, training_gender, training_classes)
        n_neighbors_ = grid.best_params_['knn__n_neighbors']

        knn_gender_classifier = knn.fit(training_gender_vectors, ttp_train)
        knn_gender_predictions = knn.predict(test_gender_vectors)
        print("K NEAREST NEIGHBORS f1-micro:", f1_score(ttp_test, knn_gender_predictions, average='micro'))
        print("K NEAREST NEIGHBORS f1-macro:", f1_score(ttp_test, knn_gender_predictions, average='macro'))
        print("K NEAREST NEIGHBORS accuracy:", accuracy_score(ttp_test, knn_gender_predictions))
        print(n_neighbors_)  # optimal n_neighbors = 3

        # LOGISTIC REGRESSION CLASSIFIER
        lr_gender_classifier = lr.fit(training_gender_vectors, ttp_train)
        lr_gender_predictions = lr.predict(test_gender_vectors)
        print("LOGISTIC REGRESSION f1-micro:", f1_score(ttp_test, lr_gender_predictions, average='micro'))
        print("LOGISTIC REGRESSION f1-macro:", f1_score(ttp_test, lr_gender_predictions, average='macro'))
        print("LOGISTIC REGRESSION accuracy:", accuracy_score(ttp_test, lr_gender_predictions))

        # NEURAL NETWORK
        nn_gender_classifier = nn.fit(training_gender_vectors, ttp_train)
        nn_gender_predictions = nn.predict(test_gender_vectors)
        print("NEURAL NETWORK f1-micro:", f1_score(ttp_test, nn_gender_predictions, average='micro'))
        print("NEURAL NETWORK f1-macro:", f1_score(ttp_test, nn_gender_predictions, average='macro'))
        print("NEURAL NETWORK accuracy:", accuracy_score(ttp_test, nn_gender_predictions))

        # VOTING CLASSIFIER
        ens = VotingClassifier(estimators=[('rf', rf), ('gnb', gnb), ('svm_unop', svm_unop)], voting='hard')
        ens_gender_classifier = ens.fit(training_gender_vectors, ttp_train)
        ens_gender_predictions = ens.predict(test_gender_vectors)
        print("VOTING ENSEMBLE f1-micro:", f1_score(ttp_test, ens_gender_predictions, average='micro'))
        print("VOTING ENSEMBLE f1-macro:", f1_score(ttp_test, ens_gender_predictions, average='macro'))
        print("VOTING ENSEMBLE accuracy:", accuracy_score(ttp_test, ens_gender_predictions))

    # FEATURE: CLUSTERS
        print("-------CLUSTERS-------")
    skf.get_n_splits(training_clusters, training_classes)
    for train_index, test_index in skf.split(training_clusters, training_classes):
        training_cluster_train = map(training_clusters.__getitem__, train_index)
        training_cluster_dev = map(training_clusters.__getitem__, test_index)

        ttp_train, ttp_test = training_classes[train_index], training_classes[test_index]
        clustervectorizer = CountVectorizer(ngram_range=(1, 1), max_features=10000, stop_words=None, token_pattern=r"(?u)\b\w+\b")

        training_cluster_vectors = clustervectorizer.fit_transform(training_cluster_train).toarray()
        test_cluster_vectors = clustervectorizer.transform(training_cluster_dev).toarray()

        print(".......")
        # NAIVE BAYES CLASSIFIER
        gnb_cluster_classifier = gnb.fit(training_cluster_vectors, ttp_train)
        gnb_cluster_predictions = gnb.predict(test_cluster_vectors)
        print("NAIVE BAYES f1-micro:", f1_score(ttp_test, gnb_cluster_predictions, average='micro'))
        print("NAIVE BAYES f1-macro:", f1_score(ttp_test, gnb_cluster_predictions, average='macro'))
        print("NAIVE BAYES accuracy:", accuracy_score(ttp_test, gnb_cluster_predictions))

        # UNOPTIMIZED SVM CLASSIFIER
        svm_unop_cluster_classifier = svm_unop.fit(training_cluster_vectors, ttp_train)
        svm_unop_cluster_predictions = svm_unop.predict(test_cluster_vectors)
        print("UNOPTIMIZED SVM f1-micro:", f1_score(ttp_test, svm_unop_cluster_predictions, average='micro'))
        print("UNOPTIMIZED SVM f1-macro:", f1_score(ttp_test, svm_unop_cluster_predictions, average='macro'))
        print("UNOPTIMIZED SVM accuracy:", accuracy_score(ttp_test, svm_unop_cluster_predictions))

        # RANDOM FOREST CLASSIFIER
        rf_cluster_classifier = rf.fit(training_cluster_vectors, ttp_train)
        rf_cluster_predictions = rf.predict(test_cluster_vectors)
        print("RANDOM FOREST f1-micro:", f1_score(ttp_test, rf_cluster_predictions, average='micro'))
        print("RANDOM FOREST f1-macro:", f1_score(ttp_test, rf_cluster_predictions, average='macro'))
        print("RANDOM FOREST accuracy:", accuracy_score(ttp_test, rf_cluster_predictions))

        # K NEAREST NEIGHBORS CLASSIFIER
        grid_params = {
            'knn__n_neighbors': [1, 2, 3, 4, 5],
        }
        folds = 10
        pipeline = Pipeline(steps=[('vec', vectorizer), ('knn', knn)])
        grid = grid_search_hyperparam_space(grid_params, pipeline, folds, training_clusters, training_classes)
        n_neighbors_ = grid.best_params_['knn__n_neighbors']

        knn_cluster_classifier = knn.fit(training_cluster_vectors, ttp_train)
        knn_cluster_predictions = knn.predict(test_cluster_vectors)
        print("K NEAREST NEIGHBORS f1-micro:", f1_score(ttp_test, knn_cluster_predictions, average='micro'))
        print("K NEAREST NEIGHBORS f1-macro:", f1_score(ttp_test, knn_cluster_predictions, average='macro'))
        print("K NEAREST NEIGHBORS accuracy:", accuracy_score(ttp_test, knn_cluster_predictions))
        print(n_neighbors_)  # optimal n_neighbors = 3

        # LOGISTIC REGRESSION CLASSIFIER
        lr_cluster_classifier = lr.fit(training_cluster_vectors, ttp_train)
        lr_cluster_predictions = lr.predict(test_cluster_vectors)
        print("LOGISTIC REGRESSION f1-micro:", f1_score(ttp_test, lr_cluster_predictions, average='micro'))
        print("LOGISTIC REGRESSION f1-macro:", f1_score(ttp_test, lr_cluster_predictions, average='macro'))
        print("LOGISTIC REGRESSION accuracy:", accuracy_score(ttp_test, lr_cluster_predictions))

        # NEURAL NETWORK
        nn_cluster_classifier = nn.fit(training_cluster_vectors, ttp_train)
        nn_cluster_predictions = nn.predict(test_cluster_vectors)
        print("NEURAL NETWORK f1-micro:", f1_score(ttp_test, nn_cluster_predictions, average='micro'))
        print("NEURAL NETWORK f1-macro:", f1_score(ttp_test, nn_cluster_predictions, average='macro'))
        print("NEURAL NETWORK accuracy:", accuracy_score(ttp_test, nn_cluster_predictions))

        # VOTING CLASSIFIER
        ens = VotingClassifier(estimators=[('rf', rf), ('gnb', gnb), ('svm_unop', svm_unop)], voting='hard')
        ens_cluster_classifier = ens.fit(training_cluster_vectors, ttp_train)
        ens_cluster_predictions = ens.predict(test_cluster_vectors)
        print("VOTING ENSEMBLE f1-micro:", f1_score(ttp_test, ens_cluster_predictions, average='micro'))
        print("VOTING ENSEMBLE f1-macro:", f1_score(ttp_test, ens_cluster_predictions, average='macro'))
        print("VOTING ENSEMBLE accuracy:", accuracy_score(ttp_test, ens_cluster_predictions))


    # FEATURE: length
        print("-------LENGTH-------")
    skf.get_n_splits(training_length, training_classes)
    for train_index, test_index in skf.split(training_length, training_classes):
        training_length_train = map(training_length.__getitem__, train_index)
        training_length_dev = map(training_length.__getitem__, test_index)

        ttp_train, ttp_test = training_classes[train_index], training_classes[test_index]

        # VECTORIZER
        vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000, token_pattern=r"(?u)\b\w+\b")
        training_length_vectors = vectorizer.fit_transform(training_length_train).toarray()
        test_length_vectors = vectorizer.transform(training_length_dev).toarray()

        print(".......")
        # NAIVE BAYES CLASSIFIER
        gnb_length_classifier = gnb.fit(training_length_vectors, ttp_train)
        gnb_length_predictions = gnb.predict(test_length_vectors)
        print("NAIVE BAYES f1-micro:", f1_score(ttp_test, gnb_length_predictions, average='micro'))
        print("NAIVE BAYES f1-macro:", f1_score(ttp_test, gnb_length_predictions, average='macro'))
        print("NAIVE BAYES accuracy:", accuracy_score(ttp_test, gnb_length_predictions))

        # UNOPTIMIZED SVM CLASSIFIER
        svm_unop_length_classifier = svm_unop.fit(training_length_vectors, ttp_train)
        svm_unop_length_predictions = svm_unop.predict(test_length_vectors)
        print("UNOPTIMIZED SVM f1-micro:", f1_score(ttp_test, svm_unop_length_predictions, average='micro'))
        print("UNOPTIMIZED SVM f1-macro:", f1_score(ttp_test, svm_unop_length_predictions, average='macro'))
        print("UNOPTIMIZED SVM accuracy:", accuracy_score(ttp_test, svm_unop_length_predictions))

        # RANDOM FOREST CLASSIFIER
        rf_length_classifier = rf.fit(training_length_vectors, ttp_train)
        rf_length_predictions = rf.predict(test_length_vectors)
        print("RANDOM FOREST f1-micro:", f1_score(ttp_test, rf_length_predictions, average='micro'))
        print("RANDOM FOREST f1-macro:", f1_score(ttp_test, rf_length_predictions, average='macro'))
        print("RANDOM FOREST accuracy:", accuracy_score(ttp_test, rf_length_predictions))

        # K NEAREST NEIGHBORS CLASSIFIER
        grid_params = {
            'knn__n_neighbors': [1, 2, 3, 4, 5],
        }
        folds = 10
        pipeline = Pipeline(steps=[('vec', vectorizer), ('knn', knn)])
        grid = grid_search_hyperparam_space(grid_params, pipeline, folds, training_length, training_classes)
        n_neighbors_ = grid.best_params_['knn__n_neighbors']

        knn_length_classifier = knn.fit(training_length_vectors, ttp_train)
        knn_length_predictions = knn.predict(test_length_vectors)
        print("K NEAREST NEIGHBORS f1-micro:", f1_score(ttp_test, knn_length_predictions, average='micro'))
        print("K NEAREST NEIGHBORS f1-macro:", f1_score(ttp_test, knn_length_predictions, average='macro'))
        print("K NEAREST NEIGHBORS accuracy:", accuracy_score(ttp_test, knn_length_predictions))
        print(n_neighbors_)  # optimal n_neighbors = 3

        # LOGISTIC REGRESSION CLASSIFIER
        lr_length_classifier = lr.fit(training_length_vectors, ttp_train)
        lr_length_predictions = lr.predict(test_length_vectors)
        print("LOGISTIC REGRESSION f1-micro:", f1_score(ttp_test, lr_length_predictions, average='micro'))
        print("LOGISTIC REGRESSION f1-macro:", f1_score(ttp_test, lr_length_predictions, average='macro'))
        print("LOGISTIC REGRESSION accuracy:", accuracy_score(ttp_test, lr_length_predictions))

        # NEURAL NETWORK
        nn_length_classifier = nn.fit(training_length_vectors, ttp_train)
        nn_length_predictions = nn.predict(test_length_vectors)
        print("NEURAL NETWORK f1-micro:", f1_score(ttp_test, nn_length_predictions, average='micro'))
        print("NEURAL NETWORK f1-macro:", f1_score(ttp_test, nn_length_predictions, average='macro'))
        print("NEURAL NETWORK accuracy:", accuracy_score(ttp_test, nn_length_predictions))

        # VOTING CLASSIFIER
        ens = VotingClassifier(estimators=[('rf', rf), ('gnb', gnb), ('svm_unop', svm_unop)], voting='hard')
        ens_length_classifier = ens.fit(training_length_vectors, ttp_train)
        ens_length_predictions = ens.predict(test_length_vectors)
        print("VOTING ENSEMBLE f1-micro:", f1_score(ttp_test, ens_length_predictions, average='micro'))
        print("VOTING ENSEMBLE f1-macro:", f1_score(ttp_test, ens_length_predictions, average='macro'))
        print("VOTING ENSEMBLE accuracy:", accuracy_score(ttp_test, ens_length_predictions))


    print("-------AGE-------")
    # FEATURE: AGE
    skf.get_n_splits(training_age_preprocessed, training_classes)
    for train_index, test_index in skf.split(training_age_preprocessed, training_classes):
        training_age_train = map(training_age_preprocessed.__getitem__, train_index)
        training_age_dev = map(training_age_preprocessed.__getitem__, test_index)


        ttp_train, ttp_test = training_classes[train_index], training_classes[test_index]

        # VECTORIZER
        vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000, token_pattern=r"(?u)\b\w+\b")
        training_age_vectors = vectorizer.fit_transform(training_age_train).toarray()
        test_age_vectors = vectorizer.transform(training_age_dev).toarray()

        print(".......")
        # NAIVE BAYES CLASSIFIER
        gnb_age_classifier = gnb.fit(training_age_vectors, ttp_train)
        gnb_age_predictions = gnb.predict(test_age_vectors)
        print("NAIVE BAYES f1-micro:", f1_score(ttp_test, gnb_age_predictions, average='micro'))
        print("NAIVE BAYES f1-macro:", f1_score(ttp_test, gnb_age_predictions, average='macro'))
        print("NAIVE BAYES accuracy:", accuracy_score(ttp_test, gnb_age_predictions))

        # UNOPTIMIZED SVM CLASSIFIER
        svm_unop_age_classifier = svm_unop.fit(training_age_vectors, ttp_train)
        svm_unop_age_predictions = svm_unop.predict(test_age_vectors)
        print("UNOPTIMIZED SVM f1-micro:", f1_score(ttp_test, svm_unop_age_predictions, average='micro'))
        print("UNOPTIMIZED SVM f1-macro:", f1_score(ttp_test, svm_unop_age_predictions, average='macro'))
        print("UNOPTIMIZED SVM accuracy:", accuracy_score(ttp_test, svm_unop_age_predictions))

        # RANDOM FOREST CLASSIFIER
        rf_age_classifier = rf.fit(training_age_vectors, ttp_train)
        rf_age_predictions = rf.predict(test_age_vectors)
        print("RANDOM FOREST f1-micro:", f1_score(ttp_test, rf_age_predictions, average='micro'))
        print("RANDOM FOREST f1-macro:", f1_score(ttp_test, rf_age_predictions, average='macro'))
        print("RANDOM FOREST accuracy:", accuracy_score(ttp_test, rf_age_predictions))

        # K NEAREST NEIGHBORS CLASSIFIER
        grid_params = {
            'knn__n_neighbors': [1, 2, 3, 4, 5],
        }
        folds = 10
        pipeline = Pipeline(steps=[('vec', vectorizer), ('knn', knn)])
        grid = grid_search_hyperparam_space(grid_params, pipeline, folds, training_age_preprocessed, training_classes)
        n_neighbors_ = grid.best_params_['knn__n_neighbors']

        knn_age_classifier = knn.fit(training_age_vectors, ttp_train)
        knn_age_predictions = knn.predict(test_age_vectors)
        print("K NEAREST NEIGHBORS f1-micro:", f1_score(ttp_test, knn_age_predictions, average='micro'))
        print("K NEAREST NEIGHBORS f1-macro:", f1_score(ttp_test, knn_age_predictions, average='macro'))
        print("K NEAREST NEIGHBORS accuracy:", accuracy_score(ttp_test, knn_age_predictions))
        print(n_neighbors_)  # optimal n_neighbors = 3

        # LOGISTIC REGRESSION CLASSIFIER
        lr_age_classifier = lr.fit(training_age_vectors, ttp_train)
        lr_age_predictions = lr.predict(test_age_vectors)
        print("LOGISTIC REGRESSION f1-micro:", f1_score(ttp_test, lr_age_predictions, average='micro'))
        print("LOGISTIC REGRESSION f1-macro:", f1_score(ttp_test, lr_age_predictions, average='macro'))
        print("LOGISTIC REGRESSION accuracy:", accuracy_score(ttp_test, lr_age_predictions))

        # NEURAL NETWORK
        nn_age_classifier = nn.fit(training_age_vectors, ttp_train)
        nn_age_predictions = nn.predict(test_age_vectors)
        print("NEURAL NETWORK f1-micro:", f1_score(ttp_test, nn_age_predictions, average='micro'))
        print("NEURAL NETWORK f1-macro:", f1_score(ttp_test, nn_age_predictions, average='macro'))
        print("NEURAL NETWORK accuracy:", accuracy_score(ttp_test, nn_age_predictions))

        # VOTING CLASSIFIER
        ens = VotingClassifier(estimators=[('rf', rf), ('gnb', gnb), ('svm_unop', svm_unop)], voting='hard')
        ens_age_classifier = ens.fit(training_age_vectors, ttp_train)
        ens_age_predictions = ens.predict(test_age_vectors)
        print("VOTING ENSEMBLE f1-micro:", f1_score(ttp_test, ens_age_predictions, average='micro'))
        print("VOTING ENSEMBLE f1-macro:", f1_score(ttp_test, ens_age_predictions, average='macro'))
        print("VOTING ENSEMBLE accuracy:", accuracy_score(ttp_test, ens_age_predictions))

    
    # APPENDED FEATURES
    skf.get_n_splits(training_texts_preprocessed, training_classes)
    for train_index, test_index in skf.split(training_texts_preprocessed, training_classes):
        training_texts_preprocessed_train = map(training_texts_preprocessed.__getitem__, train_index)
        training_texts_preprocessed_dev = map(training_texts_preprocessed.__getitem__, test_index)

        ttp_train, ttp_test = training_classes[train_index], training_classes[test_index]

        # VECTORIZER
        vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000)
        training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed_train).toarray()
        test_data_vectors = vectorizer.transform(training_texts_preprocessed_dev).toarray()

        training_gender_train = map(training_gender.__getitem__, train_index)
        training_gender_dev = map(training_gender.__getitem__, test_index)


        gender_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000, stop_words=None, token_pattern=r"(?u)\b\w+\b")

        training_gender_vectors = gender_vectorizer.fit_transform(training_gender_train).toarray()
        test_gender_vectors = gender_vectorizer.transform(training_gender_dev).toarray()


        training_data_vectors = np.concatenate((training_data_vectors, training_gender_vectors), axis=1)
        test_data_vectors = np.concatenate((test_data_vectors, test_gender_vectors), axis=1)


        training_cluster_train = map(training_clusters.__getitem__, train_index)
        training_cluster_dev = map(training_clusters.__getitem__, test_index)

        clustervectorizer = CountVectorizer(ngram_range=(1, 1), max_features=10000, stop_words=None,
                                            token_pattern=r"(?u)\b\w+\b")

        training_cluster_vectors = clustervectorizer.fit_transform(training_cluster_train).toarray()
        test_cluster_vectors = clustervectorizer.transform(training_cluster_dev).toarray()

        training_data_vectors = np.concatenate((training_data_vectors, training_cluster_vectors), axis=1)
        test_data_vectors = np.concatenate((test_data_vectors, test_cluster_vectors), axis=1)

        training_length_train = map(training_length.__getitem__, train_index)
        training_length_dev = map(training_length.__getitem__, test_index)


        length_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000, token_pattern=r"(?u)\b\w+\b")
        training_length_vectors = length_vectorizer.fit_transform(training_length_train).toarray()
        test_length_vectors = length_vectorizer.transform(training_length_dev).toarray()

        training_data_vectors = np.concatenate((training_data_vectors, training_length_vectors), axis=1)
        test_data_vectors = np.concatenate((test_data_vectors, test_length_vectors), axis=1)
        
        training_age_train = map(training_age_preprocessed.__getitem__, train_index)
        training_age_dev = map(training_age_preprocessed.__getitem__, test_index)

        age_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                     max_features=10000, token_pattern=r"(?u)\b\w+\b")
        training_age_vectors = age_vectorizer.fit_transform(training_age_train).toarray()
        test_age_vectors = age_vectorizer.transform(training_age_dev).toarray()

        training_data_vectors = np.concatenate((training_data_vectors, training_age_vectors), axis=1)
        test_data_vectors = np.concatenate((test_data_vectors, test_age_vectors), axis=1)
        
        print(".......")
        # NAIVE BAYES CLASSIFIER
        gnb = GaussianNB()
        gnb_classifier = gnb.fit(training_data_vectors, ttp_train)
        gnb_predictions = gnb.predict(test_data_vectors)
        print("NAIVE BAYES f1-micro:", f1_score(ttp_test, gnb_predictions, average='micro'))
        print("NAIVE BAYES f1-macro:", f1_score(ttp_test, gnb_predictions, average='macro'))
        print("NAIVE BAYES accuracy:", accuracy_score(ttp_test, gnb_predictions))

        # UNOPTIMIZED SVM CLASSIFIER
        svm_unop = svm.SVC(C=1, cache_size=200,
                           coef0=0.0, degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=True,
                           random_state=None, shrinking=True, tol=0.001, verbose=False)
        svm_unop_classifier = svm_unop.fit(training_data_vectors, ttp_train)
        svm_unop_predictions = svm_unop.predict(test_data_vectors)
        print("UNOPTIMIZED SVM f1-micro:", f1_score(ttp_test, svm_unop_predictions, average='micro'))
        print("UNOPTIMIZED SVM f1-macro:", f1_score(ttp_test, svm_unop_predictions, average='macro'))
        print("UNOPTIMIZED SVM accuracy:", accuracy_score(ttp_test, svm_unop_predictions))

        # RANDOM FOREST CLASSIFIER
        rf = RandomForestClassifier(n_estimators=20, random_state=1)
        rf_classifier = rf.fit(training_data_vectors, ttp_train)
        rf_predictions = rf.predict(test_data_vectors)
        print("RANDOM FOREST f1-micro:", f1_score(ttp_test, rf_predictions, average='micro'))
        print("RANDOM FOREST f1-macro:", f1_score(ttp_test, rf_predictions, average='macro'))
        print("RANDOM FOREST accuracy:", accuracy_score(ttp_test, rf_predictions))

        # K NEAREST NEIGHBORS CLASSIFIER
        grid_params = {
            'knn__n_neighbors': [1, 2, 3, 4, 5],
        }
        knn = KNeighborsClassifier()
        folds = 10
        pipeline = Pipeline(steps=[('vec', vectorizer), ('knn', knn)])
        grid = grid_search_hyperparam_space(grid_params, pipeline, folds, training_texts_preprocessed, training_classes)
        n_neighbors_ = grid.best_params_['knn__n_neighbors']

        knn_classifier = knn.fit(training_data_vectors, ttp_train)
        knn_predictions = knn.predict(test_data_vectors)
        print("K NEAREST NEIGHBORS f1-micro:", f1_score(ttp_test, knn_predictions, average='micro'))
        print("K NEAREST NEIGHBORS f1-macro:", f1_score(ttp_test, knn_predictions, average='macro'))
        print("K NEAREST NEIGHBORS accuracy:", accuracy_score(ttp_test, knn_predictions))

        # LOGISTIC REGRESSION CLASSIFIER
        lr = LogisticRegression(random_state=0)
        lr_classifier = lr.fit(training_data_vectors, ttp_train)
        lr_predictions = lr.predict(test_data_vectors)
        print("LOGISTIC REGRESSION f1-micro:", f1_score(ttp_test, lr_predictions, average='micro'))
        print("LOGISTIC REGRESSION f1-macro:", f1_score(ttp_test, lr_predictions, average='macro'))
        print("LOGISTIC REGRESSION accuracy:", accuracy_score(ttp_test, lr_predictions))

        # NEURAL NETWORK
        nn = MLPClassifier(random_state=1, max_iter=300)
        nn_classifier = nn.fit(training_data_vectors, ttp_train)
        nn_predictions = nn.predict(test_data_vectors)
        print("NEURAL NETWORK f1-micro:", f1_score(ttp_test, nn_predictions, average='micro'))
        print("NEURAL NETWORK f1-macro:", f1_score(ttp_test, nn_predictions, average='macro'))
        print("NEURAL NETWORK accuracy:", accuracy_score(ttp_test, nn_predictions))

        # VOTING CLASSIFIER
        ens = VotingClassifier(estimators=[('rf', rf), ('gnb', gnb), ('svm_unop', svm_unop)], voting='hard')
        ens_classifier = ens.fit(training_data_vectors, ttp_train)
        ens_predictions = ens.predict(test_data_vectors)
        print("VOTING ENSEMBLE f1-micro:", f1_score(ttp_test, ens_predictions, average='micro'))
        print("VOTING ENSEMBLE f1-macro:", f1_score(ttp_test, ens_predictions, average='macro'))
        print("VOTING ENSEMBLE accuracy:", accuracy_score(ttp_test, ens_predictions))






    # CONCATENATED FEATURES
    vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                max_features=10000)
    training_data_vectors = vectorizer.fit_transform(training_texts_preprocessed).toarray()
    test_data_vectors = vectorizer.transform(test_texts_preprocessed).toarray()

    gender_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                max_features=10000, stop_words=None, token_pattern=r"(?u)\b\w+\b")

    training_gender_vectors = gender_vectorizer.fit_transform(training_gender).toarray()
    test_gender_vectors = gender_vectorizer.transform(test_gender).toarray()


    training_data_vectors = np.concatenate((training_data_vectors, training_gender_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_gender_vectors), axis=1)

    clustervectorizer = CountVectorizer(ngram_range=(1, 1), max_features=10000, stop_words=None,
                                       token_pattern=r"(?u)\b\w+\b")

    training_cluster_vectors = clustervectorizer.fit_transform(training_clusters).toarray()
    test_cluster_vectors = clustervectorizer.transform(test_clusters).toarray()

    training_data_vectors = np.concatenate((training_data_vectors, training_cluster_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_cluster_vectors), axis=1)

    length_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                max_features=10000, token_pattern=r"(?u)\b\w+\b")
    training_length_vectors = length_vectorizer.fit_transform(training_length).toarray()
    test_length_vectors = length_vectorizer.transform(test_length).toarray()
    '''
    training_data_vectors = np.concatenate((training_data_vectors, training_length_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_length_vectors), axis=1)
    '''

    age_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", tokenizer=None, preprocessor=None,
                                max_features=10000, token_pattern=r"(?u)\b\w+\b")
    training_age_vectors = age_vectorizer.fit_transform(training_age_preprocessed).toarray()
    test_age_vectors = age_vectorizer.transform(test_age_preprocessed).toarray()

    training_data_vectors = np.concatenate((training_data_vectors, training_age_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_age_vectors), axis=1)

    print(".......")
    # NAIVE BAYES CLASSIFIER
    gnb = GaussianNB()
    gnb_classifier = gnb.fit(training_data_vectors, training_classes)
    gnb_predictions = gnb.predict(test_data_vectors)
    print("NAIVE BAYES f1-micro:", f1_score(test_classes, gnb_predictions, average='micro'))
    print("NAIVE BAYES f1-macro:", f1_score(test_classes, gnb_predictions, average='macro'))
    print("NAIVE BAYES accuracy:", accuracy_score(test_classes, gnb_predictions))

    # UNOPTIMIZED SVM CLASSIFIER
    svm_unop = svm.SVC(C=1, cache_size=200,
                      coef0=0.0, degree=3, gamma='auto', kernel='linear', max_iter=-1, probability=True,
                      random_state=None, shrinking=True, tol=0.001, verbose=False)
    svm_unop_classifier = svm_unop.fit(training_data_vectors, training_classes)
    svm_unop_predictions = svm_unop.predict(test_data_vectors)
    print("UNOPTIMIZED SVM f1-micro:", f1_score(test_classes, svm_unop_predictions, average='micro'))
    print("UNOPTIMIZED SVM f1-macro:", f1_score(test_classes, svm_unop_predictions, average='macro'))
    print("UNOPTIMIZED SVM accuracy:", accuracy_score(test_classes, svm_unop_predictions))

    # RANDOM FOREST CLASSIFIER
    rf = RandomForestClassifier(n_estimators=20, random_state=1)
    rf_classifier = rf.fit(training_data_vectors, training_classes)
    rf_predictions = rf.predict(test_data_vectors)
    print("RANDOM FOREST f1-micro:", f1_score(test_classes, rf_predictions, average='micro'))
    print("RANDOM FOREST f1-macro:", f1_score(test_classes, rf_predictions, average='macro'))
    print("RANDOM FOREST accuracy:", accuracy_score(test_classes, rf_predictions))

    # K NEAREST NEIGHBORS CLASSIFIER
    grid_params = {
       'knn__n_neighbors': [1, 2, 3, 4, 5],
    }
    knn = KNeighborsClassifier()
    folds = 10
    pipeline = Pipeline(steps=[('vec', vectorizer), ('knn', knn)])
    grid = grid_search_hyperparam_space(grid_params, pipeline, folds, training_texts_preprocessed, training_classes)
    n_neighbors_ = grid.best_params_['knn__n_neighbors']

    knn_classifier = knn.fit(training_data_vectors, training_classes)
    knn_predictions = knn.predict(test_data_vectors)
    print("K NEAREST NEIGHBORS f1-micro:", f1_score(test_classes, knn_predictions, average='micro'))
    print("K NEAREST NEIGHBORS f1-macro:", f1_score(test_classes, knn_predictions, average='macro'))
    print("K NEAREST NEIGHBORS accuracy:", accuracy_score(test_classes, knn_predictions))

    # LOGISTIC REGRESSION CLASSIFIER
    lr = LogisticRegression(random_state=0)
    lr_classifier = lr.fit(training_data_vectors, training_classes)
    lr_predictions = lr.predict(test_data_vectors)
    print("LOGISTIC REGRESSION f1-micro:", f1_score(test_classes, lr_predictions, average='micro'))
    print("LOGISTIC REGRESSION f1-macro:", f1_score(test_classes, lr_predictions, average='macro'))
    print("LOGISTIC REGRESSION accuracy:", accuracy_score(test_classes, lr_predictions))

    # NEURAL NETWORK
    nn = MLPClassifier(random_state=1, max_iter=300)
    nn_classifier = nn.fit(training_data_vectors, training_classes)
    nn_predictions = nn.predict(test_data_vectors)
    print("NEURAL NETWORK f1-micro:", f1_score(test_classes, nn_predictions, average='micro'))
    print("NEURAL NETWORK f1-macro:", f1_score(test_classes, nn_predictions, average='macro'))
    print("NEURAL NETWORK accuracy:", accuracy_score(test_classes, nn_predictions))

    # VOTING CLASSIFIER
    ens = VotingClassifier(estimators=[('rf', rf), ('gnb', gnb), ('svm_unop', svm_unop)], voting='hard')
    ens_classifier = ens.fit(training_data_vectors, training_classes)
    ens_predictions = ens.predict(test_data_vectors)
    print("VOTING ENSEMBLE f1-micro:", f1_score(test_classes, ens_predictions, average='micro'))
    print("VOTING ENSEMBLE f1-macro:", f1_score(test_classes, ens_predictions, average='macro'))
    print("VOTING ENSEMBLE accuracy:", accuracy_score(test_classes, ens_predictions))

