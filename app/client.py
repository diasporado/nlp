import requests
import json

url = 'http://localhost:8001/get_word_embeddings'
payload = {
    'documents': [
        ['Indonesia', 'clerical', 'body'],
        ['MUI', 'is', 'often'],
        'Indonesia’s top clerical body, the Indonesian Council of Ulama (link is external) (MUI), recently called on the government to issue legislation banning lesbian, gay, bisexual and transgender activities in Indonesia (link is external). In doing so, it referred to a religious opinion, or fatwa, published by the organisation in 2014, which found that LGBT activities were “against Islam”. Last month, MUI also issued a fatwa against that the Fajar Nusantara Movement (Gafatar), declaring that it was a deviant sect (link is external). But what do these fatwa actually mean in practice? Do fatwa issued by MUI have any real influence over policy and legal decisions in Indonesia, and how do they affect the attitudes and behaviour of Indonesian Muslims?',
        'MUI is often described as a quasi-state body. Although it receives funds from the state budget, it is, in fact, not an official government agency (contrary to the view prevailing among many Indonesians). Over recent years, it has also been able to significantly boost its operating budget through its legal monopoly in the lucrative business of halal certification and its involvement in Islamic banking (link is external). The council has a presence right across the Indonesian archipelago, with 33 branches at the provincial level and more than 400 district-level branches.',
        'MUI was established in 1975 by former President Soeharto as a means to gain Muslim support for the state’s development projects. Founded at a time when the government was anxious about the threat posed by political Islam (link is external), the council was designed to channel these interests into a forum that would not challenge the influence of the state. In its early years, MUI endorsed and promoted the national ideology of Pancasila as its ideological foundation, as Soeharto demanded. It was only in 2000 that the organisation made the formal ideological shift to Islam (link is external). While many – but not all (link is external) – of the fatwa issued by MUI during the New Order period supported state policies, in the reform era, MUI has become more independent (link is external), often challenging or attempting to influence government policy.'
    ],
    'language': 'multi',
    'type': 'document',
    'document_type': 'pooling',
    'embeddings': ['flair', 'bert', 'glove']
}

response = requests.get(url, data=json.dumps(payload), headers={'content-type':'text/plain'})