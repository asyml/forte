from termcolor import colored

from forte.data.ontology import conll03_ontology as Ont
from forte.pipeline import Pipeline

pl = Pipeline()

# Our pipeline is ready, now let's try out some text snippets.
pl.init_from_config_path("sample_config.yml")

search_engine_text = "A Scottish firm is looking to attract web surfers with " \
                     "a search engine that reads out results. Called Speegle," \
                     " it has the look and feel of a normal search engine, " \
                     "with the added feature of being able to read" \
                     " out the results. Scottish speech technology firm CEC " \
                     "Systems launched the site in November. But experts have" \
                     " questioned whether talking search engines are of any " \
                     "real benefit to people with visual impairments. The" \
                     " Edinburgh-based firm CEC has married speech " \
                     "technology with ever-popular internet search. The " \
                     "ability to search is becoming increasingly crucial to " \
                     "surfers baffled by the huge amount of information " \
                     "available on the web."

win_medal_text = "British hurdler Sarah Claxton is confident she can win her " \
                 "first major medal at next month's European Indoor " \
                 "Championships in Madrid."

pack = pl.process_one(win_medal_text)

for sentence in pack.get(Ont.Sentence):
    sent_text = sentence.text
    print(colored("Sentence:", 'red'), sent_text, "\n")

for sentence in pack.get(Ont.Sentence):
    tokens = [(token.text, token.pos_tag) for token in
              pack.get(Ont.Token, sentence)]
    print(colored("Tokens:", 'red'), tokens, "\n")
    break

for sentence in pack.get(Ont.Sentence):
    for entity in pack.get(Ont.EntityMention, sentence):
        print(colored("EntityMention:", 'red'),
              entity.text,
              'has type',
              colored(entity.ner_type, 'blue'), "\n")

for sentence in pack.get(Ont.Sentence):
    for entity in pack.get(Ont.EntityMention, sentence):
        print(f"Entity: {entity.text}")
        for token in pack.get(Ont.Token, entity):
            print(f"Has token {token.text}")
