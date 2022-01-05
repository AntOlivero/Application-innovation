from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def svm (X_train, Y_train, X_dev, Y_dev, classes_dev, X_test, classe_test, reviewId_test, ngram_range_min, ngram_range_max, nbDocCorpus) :
    svm = LinearSVC()

    classifieur = svm.fit(X_train, Y_train)

    #Résultat de notre modèle avec les données de train -> mean accuracy
    print("Score train: ", svm.score(X_train, Y_train)) 
    #Résultat de notre modèle avec les données de dev -> mean accuracy
    print("Score dev: ", svm.score(X_dev, Y_dev), "\n")

    Y_test_dev = classifieur.predict(X_dev)
    print("="*50)
    print("Résultats Corpus Dev:")
    print("="*50)
    score = classification_report(Y_dev, Y_test_dev, target_names=classes_dev)
    print(score)

    print("="*50)
    print("Prédiction Corpus Test:")
    print("="*50)
    Y_test_prediction = classifieur.predict(X_test)
    predicted_note = [classe_test[i] for i in Y_test_prediction]

    #Création du fichier résultat de sortie pour classement
    fichier_résultat = open("./out/resultat_" + str(ngram_range_min) + "-" + str(ngram_range_max) + "_" + str(nbDocCorpus) + ".txt", "w")
    for id, note in zip(reviewId_test, predicted_note):
        fichier_résultat.write(str(id) + " " + str(note) + "\n")
    fichier_résultat.close()
    print("Prédiction Terminé. Résultat sortie dans le fichier : ./out/resultat_" + str(ngram_range_min) + "-" + str(ngram_range_max) + "_" + str(nbDocCorpus) + ".txt")
    