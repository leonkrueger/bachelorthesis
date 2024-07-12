from itertools import chain

from nltk.corpus import wordnet


class SynonymGenerator:
    def get_synonyms(name: str) -> list[str]:
        """Generates synonyms for the given name and returns them in a list, including the original name"""
        raise NotImplementedError()


class WordnetSynonymGenerator(SynonymGenerator):
    def get_synonyms(name: str) -> list[str]:
        synsets = wordnet.synsets(name)
        synonyms = set([name])
        synonyms.update(
            set(
                chain.from_iterable(
                    [word.lemma_names() for word in synsets[: min(3, len(synsets))]]
                )
            )
        )
        return synonyms


class Llama3SynonymGenerator(SynonymGenerator):
    def get_synonyms(name: str) -> list[str]:
        return [name]
