from xml.dom import minidom
from html import unescape
import os
import emoji
import regex as re
import string



#Supprime les emoji de la ligne de texte passé en param
def remove_emoji(text):
    data = re.findall(r'\X', text)
    for word in data:
        if emoji.is_emoji(word):
            # Remove from the given text the emojis
            text = text.replace(word, '') 
    return text

#Supprime les URL de la ligne de texte passé en param
def remove_URL(text):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,text)
    for word in url:
        text = text.replace(word[0], '')
    return text

#Supprime la ponctuation de la ligne de texte passé en param
def remove_punctuation(s):
    table = str.maketrans(dict.fromkeys(string.punctuation))  
    new_s = s.translate(table)
    return new_s

print("Debut du parsing")
#Parse le document
xmldoc = minidom.parse('dev.xml')
print("Fin du parsing")
#Recupere les nodes note du document xml
notelist = xmldoc.getElementsByTagName('note')
#Recupere les nodes commentaire du document xml
commentairelist = xmldoc.getElementsByTagName('commentaire')
print("Debut de l'ecriture des fichier")
root = minidom.Document()
size = 10000

note_list = []
comment_list = []
if ( len(notelist) == len(commentairelist) ):
    for i in range(size):
        if(commentairelist[i].firstChild != None):

            #creer le contenue de la note
            note_list.append(notelist[i].firstChild.nodeValue)

            #creer le contenue du commentaire avec la ligne de texte sans emoji/url/ponctuation
            comment_list.append(remove_punctuation(remove_URL(remove_emoji(commentairelist[i].firstChild.nodeValue))))
            #ajoute le contenue a l'interieur de la balise commentaire

save_path_file = "notes" + str(size) + ".txt"
with open(save_path_file, "w", encoding="utf-8") as f:
    for note in note_list:
	    f.write(note + "\n")

save_path_file = "comments" + str(size) + ".txt"
with open(save_path_file, "w", encoding="utf-8") as f:
	for comment in comment_list:
	    f.write(comment + "\n")
print("Fin de l'ecriture des fichier txt")


