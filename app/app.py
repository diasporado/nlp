from flair.embeddings import FlairEmbeddings, BertEmbeddings, StackedEmbeddings, WordEmbeddings, \
    DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.data import Sentence
import spacy
import hug
import json
from helpers import correct_input_format

DEFAULT_EMBEDDING_TYPE = 'stacked'
DEFAULT_EMBEDDING_DOCUMENT_TYPE = 'pooling'
DEFAULT_EMBEDDING_LANGUAGE = 'multi'
DEFAULT_EMBEDDING_LIST = ['flair', 'bert', 'glove']

nlp = spacy.load('en_core_web_sm')
flair_embedding_forward = FlairEmbeddings('multi-forward')
flair_embedding_backward = FlairEmbeddings('multi-backward')
glove_embedding = WordEmbeddings('glove')
bert_embedding = BertEmbeddings('bert-base-multilingual-cased')

@hug.post("/get_word_embeddings", on_invalid=hug.redirect.not_found)
def get_word_embeddings(body, hug_timer=3):
    '''
    Get word embeddings
    Input: a dict with a single field
        - "documents": list of dicts, each containing a single document
        - "language": "multi", "chinese"
        - "type": "stacked", "document"
        - "document_type": "pooling" or "RNN"
        - "rnn_params": a dict consisting of {
            "hidden_size": the number of hidden states in the rnn
            "rnn_layers": the number of layers for the rnn
            "reproject_words": boolean value, indicating whether to reproject the token embeddings in a separate linear
                layer before putting them into the rnn or not
            "reproject_words_dimension": output dimension of reprojecting token embeddings. If None the same output
                dimension as before will be taken.
            "bidirectional": boolean value, indicating whether to use a bidirectional rnn or not
            "dropout": the dropout value to be used
            "word_dropout": the word dropout value to be used, if 0.0 word dropout is not used
            "locked_dropout": locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
            "rnn_type": 'GRU' or 'LSTM' or 'RNN_TANH' or 'RNN_RELU'
        }
        - "embeddings": list of embeddings to use ['flair', 'bert', 'glove']
    Output: a dict containing
        - "success": indicating if the operation is executed
        - "error": the generated error message, if any
    '''
    msg = json.loads(body)
    if not correct_input_format(msg, "documents"):
        return { 'time_taken': float(hug_timer), 'success': False, "error": "Wrong input format received"}
    
    try:
        documents = msg['documents']
        embedding_type = msg.get('type', DEFAULT_EMBEDDING_TYPE)
        embedding_document_type = msg.get('document_type', DEFAULT_EMBEDDING_DOCUMENT_TYPE)
        embedding_language = msg.get('language', DEFAULT_EMBEDDING_LANGUAGE)
        embedding_list = msg.get('embeddings', DEFAULT_EMBEDDING_LIST)
        if embedding_document_type == 'RNN':
            embedding_rnn_params = {
                'hidden_size': msg['rnn_params'].get('hidden_size', 128),
                'rnn_layers': msg['rnn_params'].get('rnn_layers', 1),
                'reproject_words': False if msg['rnn_params'].get('reproject_words', False) == 'False' else True,
                'reproject_words_dimension': msg['rnn_params'].get('reproject_words_dimension', None),
                'bidirectional': True if msg['rnn_params'].get('bidirectional', True) == 'True' else False,
                'dropout': msg['rnn_params'].get('dropout', 0.5),
                'word_dropout': msg['rnn_params'].get('word_dropout', 0.0),
                'locked_dropout': msg['rnn_params'].get('locked_dropout', 0.0),
                'rnn_type': msg['rnn_params'].get('rnn_type', 'GRU')
            }
        combined_embeddings = []

        if embedding_language == 'chinese':
            combined_embeddings.append(BertEmbeddings('bert-base-chinese'))
        else:
            if 'flair' in embedding_list:
                combined_embeddings.append(flair_embedding_forward)
                combined_embeddings.append(flair_embedding_backward)
            if 'glove' in embedding_list:
                combined_embeddings.append(glove_embedding)
            if 'bert' in embedding_list:
                combined_embeddings.append(bert_embedding)

        if embedding_type == 'stacked':
            document_embeddings = StackedEmbeddings(embeddings=combined_embeddings)
        elif embedding_type == 'document':
            if embedding_document_type == 'pooling':
                document_embeddings = DocumentPoolEmbeddings(combined_embeddings, 'mean')
            elif embedding_document_type == 'RNN':
                document_embeddings = DocumentRNNEmbeddings(combined_embeddings, **embedding_rnn_params)
        
        results = []
        for doc in documents:
            sentence = Sentence(doc)
            print(sentence)
            print(sentence.get_embedding())
            document_embeddings.embed(sentence)
            print(sentence.get_embedding())
            results.append(sentence.get_embedding())
        status = 'success'
        err = 'none'
    except:
        status = 'failed'
        err = 'error encountered'
    return { 'time_taken': float(hug_timer), 'success': status, "error": err, "results": results}

@hug.post("/get_entities", on_invalid=hug.redirect.not_found)
def get_entities(body, hug_timer=3):
    '''
    Get word embeddings
    Input: a dict with a single field
        - "documents": list of dicts, each containing a single document
        - "language": "multi", "chinese"
        - "type": "stacked", "document"
        - "document_type": "pooling" or "RNN"
        - "rnn_params": a dict consisting of {
            "hidden_size": the number of hidden states in the rnn
            "rnn_layers": the number of layers for the rnn
            "reproject_words": boolean value, indicating whether to reproject the token embeddings in a separate linear
                layer before putting them into the rnn or not
            "reproject_words_dimension": output dimension of reprojecting token embeddings. If None the same output
                dimension as before will be taken.
            "bidirectional": boolean value, indicating whether to use a bidirectional rnn or not
            "dropout": the dropout value to be used
            "word_dropout": the word dropout value to be used, if 0.0 word dropout is not used
            "locked_dropout": locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
            "rnn_type": 'GRU' or 'LSTM' or 'RNN_TANH' or 'RNN_RELU'
        }
        - "embeddings": list of embeddings to use ['flair', 'bert', 'glove']
    Output: a dict containing
        - "success": indicating if the operation is executed
        - "error": the generated error message, if any
    '''
    msg = json.loads(body)
    if not correct_input_format(msg, "documents"):
        return { 'time_taken': float(hug_timer), 'success': False, "error": "Wrong input format received"}

    documents = msg['documents']
    results = []
    for text in documents:
        doc = nlp(text)
        entities = [{'text': ent.text, 'label': ent.label} for ent in doc.ents]
    return {'time_taken': float(hug_timer), 'success': status, "error": err, "results": results}

    return {'entities': entities}