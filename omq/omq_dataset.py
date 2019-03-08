import xml.etree.ElementTree
from functional import seq

def read_data():
    categories_iter = xml.etree.ElementTree.parse('./data/OMQ/omq_public_categories.xml').getroot().iter('category')
    interactions_root = xml.etree.ElementTree.parse('./data/OMQ/omq_public_interactions.xml').getiterator('interaction')

    return categories_iter, interactions_root


def to_request_row(request_element):
    text = request_element.findtext('text/relevantText').strip()
    category = request_element.findtext('metadata/category')
    id = request_element.findtext('metadata/id')

    return {'id': id, 'category': category, 'text_raw': text }

categories, interactions = read_data()
interaction_texts = seq(interactions).map(to_request_row).map(lambda i: i['text_raw']).to_list()
texts = '\n'.join(interaction_texts)

f = open("data/omq_interactions_text.txt","w+")
for i in range(0, len(interaction_texts)):
    interaction_text = interaction_texts[i]
    f.write(interaction_text)

f.close()