import eyeball_pp
import openai


class QAAgent:
    def __init__(self):
        # Agent initialization code
        ...

    @eyeball_pp.record_task(args_to_record=["context", "question"])
    def ask(self, context: str, question: str) -> str:
        # You can write arbitrary code here, the only thing the eval framework
        # cares about is the input and output of this function.
        # The input arguments and return value is recorded

        system = """
        You are trying to answer a question strictly using the information provided in the context. Reply I don't know if you don't know the answer.
        """

        prompt = f"""
        Context: {context}
        Question: {question}
        """

        # eval params can be set when you are trying to evaluate this agent
        # with different parameters eg. different models, providers or hyperparameters like temperature
        model = eyeball_pp.get_eval_param("model") or "gpt-3.5-turbo"
        temperature = eyeball_pp.get_eval_param("temperature") or 0.5

        return "brown"
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
    eyeball_pp.set_config(
        api_key="eb26fea1b82d486b9edc58dcb882ea23", api_url="http://0.0.0.0:8081"
    )

    agent = QAAgent()
    # agent.ask(
    #     context="The quick brown fox jumps over the lazy dog",
    #     question="What color is the fox?",
    # )

    # agent.ask(
    #     context="The lazy dog which is not brown jumps over the quick brown fox",
    #     question="What color is the dog?",
    # )

    # for input_vars in eyeball_pp.rerun_recorded_examples(
    #     {"model": "gpt-4", "temperature": 0.7}
    # ):
    #     agent.ask(input_vars["context"], input_vars["question"])

    eyeball_pp.compare_recorded_checkpoints(
        task_objective="This agent tries to answer questions given a context. Verify that the agent answers the question correctly and that the answer is only based on the context.",
        num_checkpoints_per_input_hash=4,
    )
