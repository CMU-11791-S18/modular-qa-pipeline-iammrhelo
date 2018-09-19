#python quasar_pipeline.py -f CountFeaturizer -c MultinomialNaiveBayes -s count_mnb.csv
python quasar_pipeline.py -f CountFeaturizer -c SVMClassifier -s count_svm.csv
#python quasar_pipeline.py -f CountFeaturizer -c MLPClassifier -s count_mlp.csv
#python quasar_pipeline.py -f TfidfFeaturizer -c MultinomialNaiveBayes -s tfidf_mnb.csv
python quasar_pipeline.py -f TfidfFeaturizer -c SVMClassifier -s tfidf_svm.csv
#python quasar_pipeline.py -f TfidfFeaturizer -c MLPClassifier -s tfidf_mlp.csv