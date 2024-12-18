class LargeLanguageModel:
    def run_prompt(
        self, messages: list[dict[str, str]], max_new_tokens: int = 30
    ) -> str:
        """Runs a prompt on a model and returns its answer. The prompt needs to be in the messages format."""
        raise NotImplementedError()
