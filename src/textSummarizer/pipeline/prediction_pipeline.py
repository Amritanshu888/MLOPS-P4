from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:  ## This will probably take the model evaluation config
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self,text): ## Here we are writing the prediction functions
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)  ## AutoTokenizer we will be taking from the tokenizer path
        gen_kwargs = {"length_penalty":0.8, "num_beams":8, "max_length":128}  ## These are the parameters

        pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)  ## Here we have created the Pipeline

        print("Dialogue:") ## Printing the dialogue
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]  ## Then we are going to get the outputs
        print("\nModel Summary:")
        print(output)

        return output