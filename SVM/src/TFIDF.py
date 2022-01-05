from tqdm import tqdm
import re
from operator import itemgetter
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#Liste des stopwords contenus dans la lib nltk
stopwords_francais = set(stopwords.words('french'))

def tfidf(commentaire, mode, vectorizer, labelEncoder, ngram_range_min, ngram_range_max, nbDocCorpus) : 
    
    #Liste à 3 entrée contenant le text du commentaire/note commentaire/id commentaire
    commentaires_info = []

    #taille de la liste de commentaire que l'on va utiliser pour construire notre dictionnaire
    #Ne se fait pas avec le cas de test -> prend tout les commentaire du corpus test
    if mode != "test":
        if(nbDocCorpus != 0) :
            commentaire = commentaire[0:nbDocCorpus]
        

    # Pour tout les commentaires du corpus
    for com in tqdm(commentaire):

        #récupération du commentaire écrit
        commentaire_text = com.find('commentaire').text

        #récupération de l'id de la review
        commentaire_review_id = com.find('review_id').text

        # Si on est dans cas de test alors on ne connais pas la note du commentaire
        if mode == "test":
            note_commentaire = ""           
        else:
            #récupération de la note du commentaire
            note_commentaire = com.find('note').text

        #Si il y a des notes avec des commentaire écrit 
        if commentaire_text != None and note_commentaire != None:

            #lowercase tout le texte pour diminuer le nombre de token différents
            commentaire_text = commentaire_text.lower()
            #tokenize le text
            commentaire_text = re.sub(r"\s+", " ", commentaire_text, flags=re.I)
            TNRid = (commentaire_text, [note_commentaire], commentaire_review_id)

        #Si il y a des notes sans commentaire écrit
        else:
            TNRid = ("", [note_commentaire], commentaire_review_id)

        commentaires_info.append(TNRid)

    #Récupére la liste des commentaires dans une liste à part
    list_commentaire = list(map(itemgetter(0), commentaires_info))
    #Récupére la liste des notes dans une liste à part
    list_noteCommentaire = list(map(itemgetter(1), commentaires_info))
    #Récupér la liste des reviewId dans une liste à part
    list_reviewId  = list(map(itemgetter(2), commentaires_info))

    #Si on utilise le corpus de Train
    #On créé le vocabulaire utilisé avec fit
    #On réalise le calcule du tfidf avec transform
    if mode == "train":
        print("fit transform pour train")
        #TF-IDF des mots présents dans chaque commentaire dans docs (taille de doc = 5000 par défault)
        X = vectorizer.fit_transform(list_commentaire).toarray()
        Y = labelEncoder.fit_transform(list_noteCommentaire)

        #Création du fichier vocabulaire utilisé
        vocab = vectorizer.get_feature_names_out()
        output_file = open("./out/vocab_" + str(ngram_range_min) + "-" + str(ngram_range_max) + "_" + str(nbDocCorpus) + ".txt", "w")
        output_file.write("\n".join(vocab))
        output_file.close()
    #Si on utilise le corpus de Dev
    #On réalise le calcule du tfidf avec transform -> utilise le vocab créé par le fit avec le corpus de train
    elif mode == "dev":
        print("transform pour dev")
        X = vectorizer.transform(list_commentaire).toarray()
        Y = labelEncoder.transform(list_noteCommentaire)

    #Si on utilise le corpus de Test
    #On réalise le calcule du tfidf avec transform -> utilise le vocab créé par le fit avec le corpus de train
    elif mode == "test":
        print("transform pour test")
        X = vectorizer.transform(list_commentaire).toarray()
        Y = None

    #Renvoie la matrice TF-IDF des termes d'un commentaire pour tout les commentaire du corpus, 
    #   la liste des notes correspondant pour chaque commentaire,
    #   la liste des classes,
    #   la liste des id des commentaires
    return X, Y, list(labelEncoder.classes_), list_reviewId, vectorizer, labelEncoder
