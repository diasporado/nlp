from flair.embeddings import FlairEmbeddings, BertEmbeddings, StackedEmbeddings, WordEmbeddings, TokenEmbeddings, \
    DocumentPoolEmbeddings, DocumentRNNEmbeddings, DocumentEmbeddings
from flair.data import Sentence, Token
from flair.models import SequenceTagger
from typing import List, Union
import torch
import gensim
import numpy as np
import spacy
import hug
import re
import json
from helpers import correct_input_format
import os
from scipy.spatial import distance

DEFAULT_EMBEDDING_TYPE = 'document'
DEFAULT_EMBEDDING_DOCUMENT_TYPE = 'pooling'
DEFAULT_EMBEDDING_LANGUAGE = 'multi'
DEFAULT_EMBEDDING_LIST = ['flair', 'bert', 'glove']

class GensimEmbeddings(TokenEmbeddings):
    
    def __init__(self, embeddings: str, field: str = None):
        self.name: str = str(embeddings)
        self.static_embeddings = True
        self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(str(embeddings), binary=False)
        self.field = field
        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        super().__init__()
    
    def __str__(self):
        return self.name

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for i, sentence in enumerate(sentences):
            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                if 'field' not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value
                if word in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word]
                elif word.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower()]
                elif word.lower().replace(' ', '_') in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[word.lower().replace(' ', '_')]
                elif re.sub(r'[^\w\s]', '', word.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'[^\w\s]', '', word.lower())]
                else:
                    word_embedding = np.zeros(self.embedding_length, dtype='float')
                word_embedding = torch.FloatTensor(word_embedding)
                token.set_embedding(self.name, word_embedding)
        return sentences

if os.environ['USE_EMBEDDINGS'] == 'True':
    print('Loading Flair embeddings...')
    flair_embedding_forward = FlairEmbeddings('news-forward-fast')
    flair_embedding_backward = FlairEmbeddings('news-backward-fast')
    print('Loading Glove embeddings...')
    #glove_embedding = WordEmbeddings('glove')
    print('Loading Bert embeddings...')
    bert_embedding = BertEmbeddings('../root/.flair/embeddings/bert/')
    print('Loading Google News-300d embeddings...')
    #google_embedding = GensimEmbeddings('../root/.flair/embeddings/google-news/wiki-news-300d-1M-subword.vec')
    print('Loading Conceptnet embeddings...')
    #conceptnet_embedding = GensimEmbeddings('../root/.flair/embeddings/conceptnet/numberbatch-en-17.06.txt.gz')
 
elif os.environ['USE_NER'] == 'True':
    flair_ner_tagger = SequenceTagger.load('ner-ontonotes')
    nlp = spacy.load('en_core_web_sm')


def _get_word_embeddings(msg):
    results = []
    try:
        documents = msg['documents']
        embedding_type = msg.get('type', DEFAULT_EMBEDDING_TYPE)
        embedding_document_type = msg.get(
            'document_type', DEFAULT_EMBEDDING_DOCUMENT_TYPE)
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
            if 'google-news' in embedding_list:
                combined_embeddings.append(google_embedding)
            if 'conceptnet' in embedding_list:
                combined_embeddings.append(conceptnet_embedding)

        if embedding_type == 'stacked':
            document_embeddings = StackedEmbeddings(
                embeddings=combined_embeddings)
        elif embedding_type == 'document':
            if embedding_document_type == 'pooling':
                document_embeddings = DocumentPoolEmbeddings(
                    embeddings=combined_embeddings, mode='mean')
            elif embedding_document_type == 'RNN':
                document_embeddings = DocumentRNNEmbeddings(
                    combined_embeddings, **embedding_rnn_params)

        for text in documents:
            if type(text) == list:
                sentence = Sentence(use_tokenizer=False)
                for tok in text:
                    sentence.add_token(Token(tok))
            else:
                sentence = Sentence(text)
            document_embeddings.embed(sentence)
            if embedding_type == 'stacked':
                token_dict = {}
                for token in sentence:
                    token_dict[token.text] = token.embedding.numpy().tolist()
                results.append(token_dict)
            elif embedding_type == 'document':
                results.append(sentence.get_embedding().numpy().tolist())
        status = 'success'
        err = 'none'
    except:
        status = 'failed'
        err = 'error encountered'
    return {'success': status, "error": err, "results": results}


@hug.get("/get_word_embeddings", on_invalid=hug.redirect.not_found)
def get_word_embeddings(body, hug_timer=3):
    '''
    Get word embeddings
    Input: a dict with a single field
        - "documents": list of documents or list of token lists
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
        return {'time_taken': float(hug_timer), 'success': False, "error": "Wrong input format received"}
    output = _get_word_embeddings(msg)
    output['time_taken'] = float(hug_timer)
    return output
    

@hug.get("/compare_embedding_similarity", on_invalid=hug.redirect.not_found)
def compare_embedding_similarity(body, hug_timer=3):
    '''
    Compare similarity between documents using word embeddings
    Input: a dict with a single field
        - "documents": list of documents to compare
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
    if not correct_input_format(msg, 'documents'):
        return {'time_taken': float(hug_timer), 'success': False, 'error': 'Wrong input format received'}

    try:
        documents = msg['documents']
        for docs in documents:
            if len(docs) < 2:
                raise ValueError('Not enough documents to compare. The method requires a pair of documents.')
            elif len(docs) > 2:
                raise ValueError('Too many documents passed. The method requires a pair of documents.')
        new_msg = msg.copy()
        distances = []
        for docs in documents:
            embeddings = []
            for doc in docs:
                new_msg['documents'] = [doc]
                embeddings.append(_get_word_embeddings(new_msg)['results'][0])
            distances.append(distance.cosine(embeddings[0], embeddings[1]))
        return {'time_taken': float(hug_timer), 'success': True, 'distances': distances}
    except:
        {'time_taken': float(hug_timer), 'success': False, 'distance': None}


@hug.get("/get_entities", on_invalid=hug.redirect.not_found)
def get_entities(body, hug_timer=3):
    '''
    Get Named Entities
    Input: a dict
        - "documents": list of documents
        - "language"
        - "tagger": "spacy" or "flair"
    Output: a dict containing
        - "time_taken": the time taken for the inference
        - "success": indicating if the operation is executed
        - "error": the generated error message, if any
        - "results": a list of the results of the query
    '''
    msg = json.loads(body)
    if not correct_input_format(msg, "documents"):
        return {'time_taken': float(hug_timer), 'success': False, "error": "Wrong input format received"}

    results = []
    try:
        documents = msg['documents']
        ner_tagger = msg.get('tagger', 'spacy')
        results = []

        for text in documents:
            if ner_tagger == 'spacy':
                doc = nlp(text)
                entities = [{'text': ent.text, 'label': ent.label}
                            for ent in doc.ents]
            elif ner_tagger == 'flair':
                sentence = Sentence(text)
                flair_ner_tagger.predict(sentence)
                entities = [{'text': ent['text'], 'label': ent['type'], 'confidence': ent['confidence']}
                            for ent in sentence.to_dict('ner')['entities']]
            results.append(entities)
        status = 'success'
        err = 'none'
    except:
        status = 'failed'
        err = 'error encountered'
    return {'time_taken': float(hug_timer), 'success': status, "error": err, "results": results}
