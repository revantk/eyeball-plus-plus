import eyeball_pp
import openai


class QAAgent:
    def _get_context(self, query: str) -> str:
        if "dog" in query:
            return "The lazy dog which is not brown jumps over the quick brown fox"
        else:
            return "The quick brown fox jumps over the lazy dog"

    @eyeball_pp.record_task(args_to_record=["context", "question"])
    def ask(self, question: str) -> str:
        # You can write arbitrary code here, the only thing the eval framework
        # cares about is the input and output of this function.
        # The input arguments and return value is recorded

        system = """
        You are trying to answer a question strictly using the information provided in the context. Reply I don't know if you don't know the answer.
        """
        context = self._get_context(question)
        eyeball_pp.record_intermediary_state("context", context)

        prompt = f"""
        Context: {context}
        Question: {question}
        """

        # eval params can be set when you are trying to evaluate this agent
        # with different parameters eg. different models, providers or hyperparameters like temperature
        model = eyeball_pp.get_eval_param("model") or "gpt-3.5-turbo"
        temperature = eyeball_pp.get_eval_param("temperature") or 0.5

        output = openai.ChatCompletion.create(  # type: ignore
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )["choices"][0]["message"][
            "content"
        ]  # type: ignore
        return output


if __name__ == "__main__":
    eyeball_pp.set_config(dir_path="examples")
    eyeball_pp.set_config(record_in_memory=True)
    agent = QAAgent()

    agent.ask(
        question="What color is the fox?",
    )

    agent.ask(
        question="What color is the dog?",
    )

    for input_vars in eyeball_pp.rerun_recorded_examples({"temperature": 0.2}):
        agent.ask(input_vars["question"])

    # eyeball_pp.set_config(
    #     api_key="eb26fea1b82d486b9edc58dcb882ea23", api_url="http://localhost:8081"
    # )
    # eyeball_pp.set_config(api_key="1126bf63fc4d44c7bf53e9d6442ae9b9")
    # agent = QAAgent()
    # agent.ask(
    #     question="What color is the fox?",
    # )

    # agent.ask(
    #     question="What color is the dog?",
    # )

    # for input_vars in eyeball_pp.rerun_recorded_examples({"temperature": 0.2}):
    #     agent.ask(input_vars["question"])

    eyeball_pp.evaluate_system(
        task_objective="This agent tries to answer questions given a context. Verify that the agent answers the question correctly and that the answer is only based on the context.",
        intermediate_objectives={
            "context": "The context should be relevant to the question",
        },
        num_checkpoints_per_input_hash=4,
    )
    # eyeball_pp.calculate_system_health()
