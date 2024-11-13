from itertools import chain

from nltk.corpus import wordnet

from ..large_language_model import LargeLanguageModel


class SynonymGenerator:
    def get_synonyms(self, name: str) -> list[str]:
        """Generates synonyms for the given name and returns them in a list, including the original name"""
        raise NotImplementedError()


class WordnetSynonymGenerator(SynonymGenerator):
    def get_synonyms(self, name: str) -> list[str]:
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


class LLMSynonymGenerator(SynonymGenerator):
    def __init__(self, model: LargeLanguageModel) -> None:
        self.model = model

    def get_synonyms(self, name: str) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": "Generate 5 synonyms for the given database identifier. Separate them with a comma. "
                "Answer only with the synonyms. Don't give any explanation for your result.",
            },
            {
                "role": "user",
                "content": f"Generate synonyms for '{name}'\n" "Synonyms:",
            },
        ]
        synonyms = set([name])
        synonyms.update(
            [
                synonym.strip().replace(" ", "_")
                for synonym in self.model.run_prompt(messages, 50).split(",")
            ]
        )
        return synonyms
