def sentiment_score_check(text):
    """
    Description:
        Analyze the sentiment of a given text using the VADER Sentiment Analysis tool from the NLTK library.
        Analyzes the sentiment of a given text. 
        This function determines whether the text has a negative sentiment or potentially sarcasm (indicated by a positive sentiment).

    Arguments:
        text: str - The text string whose sentiment is to be analyzed.

    Returns:
        bool - Returns False if the text has negative sentiment (indicating no need for sarcasm translation).
               Returns True if the text has positive sentiment, suggesting potential sarcasm.
    """

    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Sentiment analysis with VADER
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)

    # If sentiment score is negative, it means the tect is already expressing negative emotions.
    if scores['pos'] <= scores['neg']:
        print("Negative Sentiment Scores, No Need To Translate")
        return False
    # If the score is positive, it means there must be something sarcasm
    else:
        return True

def sentiment_score_adjective(text):
    """
    Description:
        Analyzes a given text to extract and evaluate the sentiment of adjectives. 
        This function tokenizes the text, removes stopwords, performs POS (part-of-speech) tagging. 
        Then scores each adjective using the Afinn Sentiment Analysis tool.
        The Afinn Sentiment Analysis tool scores each adjective. Only adjectives with a non-negative sentiment score 

    Arguments:
        text: str - The text string to be analyzed for adjectives and their sentiment.

    Returns:
        list of str - A list of adjectives from the text that have a non-negative sentiment score.
    """

    from afinn import Afinn
    import string
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.corpus import stopwords

    # Remove punctutation from the sentence
    text_clean = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    stop_words = stopwords.words('english')  # Get the set of English stopwords
    add = ['mr', 'mrs', 'wa', 'dr', 'said', 'back', 'could', 'one', 'looked',
           'know', 'around', 'dont', 'i', 'me', 'he', 'him', 'she', 'her', 'we']
    for sp in add:
        stop_words.append(sp)

    # Remove stop words
    tokens_cleaned = [word for word in tokens if word.lower()
                      not in stop_words]

    # Get POS tagging
    pos_tags = pos_tag(tokens_cleaned)

    # Find the sentiment_score for each adjective and append the positive one
    adj_word = []
    for w in pos_tags:
        if w[1] in ['JJ', 'JJR', 'JJS']:
            afinn = Afinn()
            sentiment_score = afinn.score(w[0])
            if sentiment_score >= 0:
                adj_word.append(w[0])

    return adj_word

def antonyms_for(word):
    """
    Description:
        NLTK uses the WordNet thesaurus to retrieve antonyms for a given word. 
        It iterates the word's synonym set (cognitive synonym set), finds the lemmas. 
        Then collects their antonyms if they are categorized as adjectives.

    Arguments:
        word: str - The word for which antonyms are to be found.

    Returns:
        list of str - A list of antonym words, specifically adjectives, for the given word.
    """
    from nltk.corpus import wordnet
    antonyms = set()
    for ss in wordnet.synsets(word):
        for lemma in ss.lemmas():
            any_pos_antonyms = [antonym.name() for antonym in lemma.antonyms()]
            for antonym in any_pos_antonyms:
                antonym_synsets = wordnet.synsets(antonym)
                if wordnet.ADJ not in [ss.pos() for ss in antonym_synsets]:
                    continue
                antonyms.add(antonym)
    return list(antonyms)

def adjective_translation(text, adj_list):
    """
    "adjective_translation"

    Description:
        Translate the positive adjectives in a given text into antonyms. 
        For each adjective in the list, find its antonym, rate them using Afinn. 
        Then choose the antonym that is closest to the original adjective's sentiment score.

    Arguments:
        text: str - The original text containing adjectives to be translated.
        adj_list: list of str - A list of adjectives identified in the text to be translated.

    Returns:
        list of str - A list of text variations, each with one of the adjectives replaced by its 
                      antonym. If no suitable antonyms are found for any of the adjectives, it may
                      return an empty list or a message indicating failure.
    """

    from afinn import Afinn
    adj_dictionary = {}

    # For each positive adjective, find the antonym for that word
    for adj in adj_list:
        antonyms = antonyms_for(adj)
        # If no antonyms returned, continue to next word
        if len(antonyms) == 0:
            print(f'"{adj}" is detected to be a potential sarcasm word, but no antonyms translated for "{adj}", continue to next adjective')
            continue
        antonyms_score = {}
        for antonym in antonyms:
            afinn = Afinn()
            antonym_score = afinn.score(antonym)
            word_score = afinn.score(adj)
            total_score = abs(antonym_score+word_score)
            antonyms_score[antonym] = total_score
        # Choose the antonym with the closest opposite afinn score to the original
        key_with_max_value = min(antonyms_score, key=antonyms_score.get)
        adj_dictionary[adj] = key_with_max_value

    # Loop through each adjective in the dicitonary and replace the orignial text string
    non_sarcasm_candidates = []
    for key, value in adj_dictionary.items():
        text_edit = text.replace(key, value)
        non_sarcasm_candidates.append(text_edit)

    if len(non_sarcasm_candidates) == 0:
        print(
            "translation failed, unable to find a non-sarcasm verison of the original text")
        return

    return non_sarcasm_candidates


# text = "well, there ya go! that explains everything gansao, we have been sincerely debating a serious issue of human development with a simpleton who gets his education from fictional tv. now i understand completely why it has been impossible to reason with him. we are officially wasting our time even communicating with such a person. emoticonxwow emoticonxdonno emoticonxfrazzled"
# if sentiment_score_check(text):
#     adj_list = sentiment_score_adjective(text)
#     print(adj_list)
#     non_sarcasm_candidates = adjective_translation(text, adj_list)
# non_sarcasm_candidates



