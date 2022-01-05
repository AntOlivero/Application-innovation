from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from TFIDF import *
from TFIDFPol import *
from SVM import *
import argparse
from lxml import etree
import time

# argument de lancement du programme 
parser = argparse.ArgumentParser(description='Multilabel Classification')
parser.add_argument('--input', type=str, default="./corpus/", help='Chemain entrée de nos données')
parser.add_argument('--ngram_range_min', type=int, default=1, help='Limite basse de la taille du ngram qui sera utilisé pour créer le dictionnaire')
parser.add_argument('--ngram_range_max', type=int, default=1, help='Limite haute de la taille du ngram qui sera utilisé pour créer le dictionnaire')
parser.add_argument('--size_dic', type=int, default=33298, help='Nombre de commentaire utilisé pour le TF-IDF des corpus Train et Dev')
parser.add_argument('--polarised', type=bool, default=False, help='Réalise le TFIDF avec le dictionnaire de mot polarisé')
args = parser.parse_args()

#Ce chargera de transformer nos phrases en matrices TFIDF
vectorizer = TfidfVectorizer(
    analyzer='word',
    #max_features=15000, #Prend encompte X mots les plus utilisé 
    max_df=0.99, #Ignore les termes ayant une df supérieur à 0,99 pour construire le vocabulaire
    min_df=0.009, #Ignore les termes aynat une df inférieur à 0,001 pour construire le vocabulaire
    lowercase=True, #Converti tout les caratères en lowecase avant de les tokenizer
    stop_words = stopwords_francais, #Stop word que l'on décide d'ignorer pour le TFIDF
    ngram_range=(args.ngram_range_min, args.ngram_range_max), #Range du Ngram que l'on créé dans le vocabulaire 
    strip_accents="unicode",
)
#Créé nos classes en fonctions des notes qu'il trouvera dans la liste des notes
labelEncoder = LabelEncoder()

 

start = time.time()

#Chargement des commentraires du fichier train.xml dans une variable commentaireTrain
print("Chargement de " + args.input + "train.xml")
#Création d'un arbre de donnée à partir du document train.xml
TRAIN = etree.parse(args.input + "train.xml")
#Récupére les informations contenu dans les balises comment
commentaireTrain = TRAIN.xpath("//comment")
print("Fichier Train chargé")
print("Nombre de commentaire dans le corpus train : ", len(commentaireTrain), "\n")

#Chargement des commentaires du fichier dev.xml dans une variable commentaireDev
print("Chargement de " + args.input + "dev.xml")
#Création d'un arbre de donnée à partir du document dev.xml
DEV  = etree.parse(args.input + "dev.xml")
#Récupére les informations contenu dans les balises comment
commentaireDev = DEV.xpath("//comment")
print("Fichier Dev chargé!")
print("Nombre de commentaire dans le corpus dev : ", len(commentaireDev), "\n")

#Chargement des commentaires du fichier test.xml dans une variable contentTest
print("Chargement de " + args.input + "test.xml")
#Création d'un arbre de donnée à partir du document test.xml
TEST  = etree.parse(args.input + "test.xml")
#Récupére les informations contenu dans les balises comment
commentaireTest = TEST.xpath("//comment")
print("Fichier Test chargé!")
print("Nombre de commentaire dans le corpus de Test : ", len(commentaireTest), "\n")

print("Tout les corpus ont été chargé")

if(args.polarised == False) :
    print("\n TFIDF des commentaire sans dictionnaire polarisé")
    print("\n TF-IDF pour les données de train")
    X_train, Y_train, classes_train, reviewId_train, vectorizer, labelEncoder = tfidf(commentaireTrain, "train", vectorizer, labelEncoder, args.ngram_range_min, args.ngram_range_max, args.size_dic)

    print("\n TF-IDF pour les données de dev")
    X_dev, Y_dev, classes_dev, reviewId_dev, vectorizer, labelEncoder = tfidf(commentaireDev, "dev", vectorizer, labelEncoder, args.ngram_range_min, args.ngram_range_max, args.size_dic)

    print("\n TF-IDF pour les données de test")
    X_test, Y_test, classes_test, reviewId_test, vectorizer, labelEncoder = tfidf(commentaireTest, "test", vectorizer, labelEncoder, args.ngram_range_min, args.ngram_range_max, args.size_dic)
else :
    print("\n TFIDF des commentaire avec dictionnaire polarisé")
    print("\n TF-IDF pour les données de train")
    X_train, Y_train, classes_train, reviewId_train, vectorizer, labelEncoder = tfidfpol(commentaireTrain, "train", vectorizer, labelEncoder, args.ngram_range_min, args.ngram_range_max, args.size_dic)

    print("\n TF-IDF pour les données de dev")
    X_dev, Y_dev, classes_dev, reviewId_dev, vectorizer, labelEncoder = tfidfpol(commentaireDev, "dev", vectorizer, labelEncoder, args.ngram_range_min, args.ngram_range_max, args.size_dic)

    print("\n TF-IDF pour les données de test")
    X_test, Y_test, classes_test, reviewId_test, vectorizer, labelEncoder = tfidfpol(commentaireTest, "test", vectorizer, labelEncoder, args.ngram_range_min, args.ngram_range_max, args.size_dic)

svm(X_train, Y_train, X_dev, Y_dev, classes_dev, X_test, classes_test, reviewId_test, args.ngram_range_min, args.ngram_range_max, args.size_dic)

end = time.time()
print("temps d'exécution : " + str(end-start))
