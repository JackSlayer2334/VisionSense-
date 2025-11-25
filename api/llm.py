import os
from .config import LLM_PROVIDER

class LLMClient:
    def __init__(self):
        self.key = os.getenv("OPENAI_API_KEY")
        if self.key:
            import openai
            openai.api_key = self.key
            self.client = openai
        else:
            self.client = None

    def describe(self, detections, texts):
        objects = detections["labels"]

        if self.client is None:
            return f"Objects seen: {objects}. Text: {texts}"

        prompt = f"""
        Describe this scene for a blind user.
        Objects detected: {objects}
        Text found: {texts}
        Provide a short helpful audio description.
        """

        res = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=80
        )

        return res.choices[0].message.content
