import argparse
import re
import pickle as pk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Fonction de prétraitement
def preprocess_text(text):
    # Mise en minuscules
    text = text.lower()

    # Suppression des caractères spéciaux et des chiffres
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenisation
    tokens = word_tokenize(text)

    # Suppression des mots vides (stopwords)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Recomposition du texte prétraité
    return ' '.join(tokens)

# Charger le modèle et le vectoriseur
def load_model(model_path, vectorizer_path):
    # Ouvre le fichier du modèle en mode binaire pour le charger
    with open(model_path, 'rb') as model_file:
        model = pk.load(model_file)  # Charge le modèle

    # Ouvre le fichier du vectoriseur en mode binaire pour le charger
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pk.load(vectorizer_file)  # Charge le vectoriseur
    return model, vectorizer

# Prétraiter et classifier la phrase
def classify_sentence(sentence, model, vectorizer):
    # Appliquer le prétraitement avant la vectorisation
    preprocessed_sentence = preprocess_text(sentence)

    # Transformer la phrase prétraitée en vecteur TF-IDF
    sentence_vector = vectorizer.transform([preprocessed_sentence])

    # Prédire la classe
    prediction = model.predict(sentence_vector)
    return prediction

# Afficher le résultat de classification
def display_result(prediction):
    print("Prédiction : ", prediction)

# Fonction principale pour gérer les options CLI
def main():
    parser = argparse.ArgumentParser(description="Classificateur de phrases avec régression logistique")
    parser.add_argument('-m', '--model', type=str, required=True, help="Chemin vers le modèle pré-entrainé")
    parser.add_argument('-v', '--vectorizer', type=str, required=True, help="Chemin vers le vectoriseur TF-IDF")
    args = parser.parse_args()

    # Charger le modèle et le vectoriseur
    model, vectorizer = load_model(args.model, args.vectorizer)

    # Entrée de l'utilisateur
    print("Entrez une phrase en anglais pour la classifier (Ctrl+D pour quitter) :")
    try:
        while True:
            sentence = input("> ").strip()
            if sentence:
                # Classifier la phrase
                prediction = classify_sentence(sentence, model, vectorizer)
                display_result(prediction)
    except EOFError:
        print("\nFin de l'entrée.")

if __name__ == "__main__":
    main()