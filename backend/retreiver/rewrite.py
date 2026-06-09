import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GROQ_API_KEY")
class QueryRewriter:
    def __init__(self):
        # Initialize the Groq client using your configured API key
        self.client = Groq(api_key=api_key)
        self.model = "llama3-8b-8192"

    def generate_search_queries(self, raw_query: str) -> list[str]:
        """
        Intercepts the raw user query and expands it into 3 distinct search vectors
        to maximize dense and sparse retrieval coverage.
        """
        system_prompt = (
            "You are an expert AI research assistant optimizing queries for a Hybrid RAG system. "
            "Your job is to take a raw user query regarding computer science literature and expand it "
            "into three distinct search strings optimized for vector search and keyword indices.\n\n"
            "Generate exactly 3 variations using these strict guidelines:\n"
            "Line 1: The original raw user query unchanged.\n"
            "Line 2: A semantic expansion focused on abstract conceptual meaning, synonyms, and theoretical terms.\n"
            "Line 3: A keyword expansion focused purely on technical terms, mathematical components, identifiers, or loss functions.\n\n"
            "Rules:\n"
            "- Output EXACTLY three lines.\n"
            "- Do not add any numbering, markdown bullets, prefixes (like 'Line 1:'), or explanatory prose.\n"
            "- Keep each expanded query concise and relevant to the domain of academic research papers."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Raw Query: {raw_query}"}
                ],
                temperature=0.2,  # Low temperature for deterministic, structured output
                max_tokens=150
            )
            
            # Extract content and split cleanly by lines
            raw_output = response.choices[0].message.content.strip()
            queries = [line.strip() for line in raw_output.split("\n") if line.strip()]
            
            # Strip out accidental LLM numbers or prefixes if they leak through
            cleaned_queries = []
            for q in queries[:3]:
                cleaned = re.sub(r'^(Line\s*\d+:|\d+\.\s*|-)\s*', '', q, flags=re.IGNORECASE)
                cleaned_queries.append(cleaned)
                
            # Fallback policy: if the LLM output is malformed, always preserve at least the raw query
            if not cleaned_queries:
                return [raw_query]
                
            return cleaned_queries

        except Exception as e:
            print(f"Query rewriting layer failed: {str(e)}. Falling back to raw query.")
            return [raw_query]

# Simple execution test block
if __name__ == "__main__":
    rewriter = QueryRewriter()
    test_query = "what loss function is used for regularization"
    expanded = rewriter.generate_search_queries(test_query)
    
    print("\n--- QUERY EXPANSION AUDIT ---")
    print(f"Original Raw Input: '{test_query}'\n")
    for idx, q in enumerate(expanded, 1):
        print(f"Search Vector {idx}: {q}")