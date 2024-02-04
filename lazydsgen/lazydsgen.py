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


class LazyDSGenerator():

    def __init__(self, input_data, 
                 desc: str, 
                 model_name:str = "gpt-3.5-turbo",
                 temmperature : float = 0.25,
                 n_tries: int = 4):
        self.input_data = input_data
        self.desc = desc
        self.llm = ChatOpenAI(model_name = model_name, 
                              temperature = temmperature)
        self._function_file_name = "generated.py"
        self.n_tries = 4
        openai.api_key = os.environ['OPENAI_API_KEY']
        
    def _clean_cache(self):
        try:
            os.remove(self._function_file_name)
        except:
            pass
        try:
            del sys.modules['generated']
        except:
            pass
    
    def generate_query(self, text):
        prompt = f"The user has asked the following question: \n{text}\n\n" 
        prompt += "DESCRIPTION OF THE DATA:\n" + self.desc
        prompt += TEMPLATE_PROMPT_INSTRUCTIONS.replace("TYPE_OF_DATA",str(type(self.input_data)))
        return prompt

    def run_generated_function(self):
        import generated
        answer = generated.fun_generated(self.input_data)
        return answer
        
    def invoke_with_tries(self, question, verbose = False, return_metadata = True):
        self._clean_cache()
        prompt = self.generate_query(question)
        t = 0
        while t < self.n_tries:
            try:
                code_generated = self.llm.predict(prompt)
                if verbose:
                    print(code_generated)

                # Open the file in write mode and save the string
                with open(self._function_file_name, 'w') as file:
                    file.write(code_generated.replace("```python","").replace("```",""))

                result = self.run_generated_function()        

                answer = {"result": result}
                if return_metadata:
                    answer["code"] = code_generated
                    answer["input"] = question
                    answer["prompt"] = prompt

                self._clean_cache()

                return answer
            except:
                t += 1
                print(f"Solution failed for n_tries={t}, try another one...")
        return {"result":"I failed at my mission :("}

    def invoke(self, question, verbose = False, return_metadata = True):
        self._clean_cache()
        prompt = self.generate_query(question)
        
        code_generated = self.llm.predict(prompt)
        if verbose:
            print(code_generated)

        # Open the file in write mode and save the string
        with open(self._function_file_name, 'w') as file:
            file.write(code_generated.replace("```python","").replace("```",""))

        result = self.run_generated_function()        

        answer = {"result": result}
        if return_metadata:
            answer["code"] = code_generated
            answer["input"] = question
            answer["prompt"] = prompt

        self._clean_cache()

        return answer