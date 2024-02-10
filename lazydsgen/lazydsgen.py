import os
import sys
import openai
from langchain.chat_models import ChatOpenAI

TEMPLATE_PROMPT_INSTRUCTIONS = """YOUR TASK:
Generate a function called `fun_generated` that takes the input 'data' (that is a TYPE_OF_DATA)) 
and solves the problem that the user is asking.
Only write the code without any other comment or explanation.
        
Include the imports neccesary to run the code.
        
THE CODE:
"""

TEMP_FUNCTION_FILE = "generated.py"


class LazyDSGenerator:
    """
    LazyDSGenerator is a class that helps you to do to run automatically simple task on an input dataframe
    with the help of GPT coding habilities. It will automatically generate the code to complete the task
    and run it. It will display or return whatever the LLM generated function does.

    You'll need to have the OpenAI API key as an env variable in OPENAI_API_KEY in order to use this class.

    Author: Ignacio Ordovás Pascual <ordovaspascual@gmail.com>

    Parameters
    ----------
    fit_intercept :
        The input data set (could be a dataframe or something similar).
    desc : str
        A description of what is inside the input data and the different features.
    (Optional) model_name : str = "gpt-3.5-turbo" :
        GPT model to run the code. By default it is selected "gpt-3.5-turbo" but
        "gpt-4" is more effective.
    (Optional) temmperature : float = 0.25 :
        Temperaure of the GPT model.
    (Optional) n_tries : int = 4 :
        Number of times "invoke_with_tries" will try to run the code in case it is
        failing to generate a runnable code.

    Parameters
    ----------
        invoke : Input a query with the instruction to be performed.
            text : str
                The prompt with the instruction.
            verbose : Bool = False
                If True it prints the code that is generated to run the task.
            return_metadata : Bool = True
                See Output

        invoke_with_tries : Input a query with the instruction to be performed, but if it fails
            it will retry up to n_tries. If it fails for all n_tries it will return an string
            saying that the task failed. The input parameters are the same as invoke.


    Output
    ------
        It will return an dictionary with the key "result" with the output of the LLM generated function.
        If return_metadata = True it will also return the input question, the promt that is sent to the LLM
        and the generated code to run the task.

    Example
    -------
        >>> dsgen = LazyDSGenerator(dataframe, "Text with a description of the data inside the dataframe and its columns")
        >>> dsgen.invoke("Make a bar plot of the number of items sold for each month.")

    """

    def __init__(
        self,
        input_data,
        desc: str,
        model_name: str = "gpt-3.5-turbo",
        temmperature: float = 0.25,
        n_tries: int = 4,
    ):
        self.input_data = input_data
        self.desc = desc
        self.llm = ChatOpenAI(model_name=model_name, temperature=temmperature)
        self.n_tries = 4
        openai.api_key = os.environ["OPENAI_API_KEY"]

    @staticmethod
    def _clean_cache_function():
        if os.path.exists(TEMP_FUNCTION_FILE):
            os.remove(TEMP_FUNCTION_FILE)
        try:
            del sys.modules["generated"]
        except:
            pass

    @staticmethod
    def clean_cache(func):
        def wrapper(self, *args, **kwargs):
            LazyDSGenerator._clean_cache_function()
            r = func(self, *args, **kwargs)
            LazyDSGenerator._clean_cache_function()
            return r

        return wrapper

    def _generate_query(self, text):
        prompt = f"The user has asked the following question: \n{text}\n\n"
        prompt += "DESCRIPTION OF THE DATA:\n" + self.desc
        prompt += TEMPLATE_PROMPT_INSTRUCTIONS.replace(
            "TYPE_OF_DATA", str(type(self.input_data))
        )
        return prompt

    def __run_generated_function(self):
        import generated

        answer = generated.fun_generated(self.input_data)
        return answer

    @clean_cache
    def invoke_with_tries(self, question, verbose=False, return_metadata=True):
        # self._clean_cache()
        prompt = self._generate_query(question)
        t = 0
        while t < self.n_tries:
            try:
                code_generated = self.llm.predict(prompt)
                if verbose:
                    print(code_generated)

                # Open the file in write mode and save the string
                with open(TEMP_FUNCTION_FILE, "w") as file:
                    file.write(
                        code_generated.replace("```python", "").replace("```", "")
                    )

                result = self.__run_generated_function()

                answer = {"result": result}
                if return_metadata:
                    answer["code"] = code_generated
                    answer["input"] = question
                    answer["prompt"] = prompt

                return answer
            except:
                t += 1
                print(f"Solution failed for n_tries={t}, try another one...")
        return {
            "result": "TASK FAILED: A succesful running code could not be generated."
        }

    @clean_cache
    def invoke(self, question, verbose=False, return_metadata=True):
        prompt = self._generate_query(question)

        code_generated = self.llm.predict(prompt)
        if verbose:
            print(code_generated)

        # Open the file in write mode and save the string
        with open(TEMP_FUNCTION_FILE, "w") as file:
            file.write(code_generated.replace("```python", "").replace("```", ""))

        result = self.__run_generated_function()

        answer = {"result": result}
        if return_metadata:
            answer["code"] = code_generated
            answer["input"] = question
            answer["prompt"] = prompt

        return answer
