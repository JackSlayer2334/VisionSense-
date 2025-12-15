class LLMClient:
    def describe(self, detections, text):
        objects = detections.get("labels", [])
        if not objects and not text:
            return "No significant objects or readable text detected."

        desc = []
        if objects:
            desc.append(f"Objects detected: {', '.join(set(objects))}.")
        if text:
            desc.append(f"Text reads: {', '.join(text)}.")

        return " ".join(desc)
