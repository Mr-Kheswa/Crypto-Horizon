#===================================================================================================
# Cell: 7 Chatbot Assistance - local agent.
# This file creates a self-contained assistant that works without any API key or external service.
# It answers project-specific questions using a local Hugging Face model when available,
#===================================================================================================

import os
from typing import List, Optional

# Import Transformers only if it is available in the environment.
# If not installed, the agent still works using the rule-based fallback responses.
try:
    from transformers import pipeline as build_local_pipeline
except Exception:  # pragma: no cover
    build_local_pipeline = None


class CryptoHorizonAgent:
    """Offline AI-style assistant for the Crypto Horizon project."""

    def __init__(self, model_name: Optional[str] = None):
        # Default to a lightweight local model; if it cannot load, the agent still works
        # through the built-in response logic that matches the project domain.
        self.model_name = model_name or "distilgpt2"
        self.model = None

        # Project metadata for the assistant so it stays grounded in the actual workflow.
        self.project_context = {
            "title": "Crypto Horizon",
            "focus": "BNB/USD and BTC/USD market structure, regime detection, correlation analysis, and LSTM forecasting",
            "data_source": "Crypto_Data_Since_2015.csv",
            "tech_stack": ["pandas", "scikit-learn", "TensorFlow/Keras", "LSTM", "Dash", "Plotly"],
            "regimes": [
                "Balanced Stability",
                "Market Indecision/Chop",
                "High-Volatility Shift",
            ],
        }

        # Load the optional local transformer model during initialization.
        self._load_local_model()

    def _load_local_model(self):
        # If Transformers is unavailable, this remains silent and the fallback logic takes over.
        if build_local_pipeline is None:
            self.model = None
            return

        try:
            # Create a local generation pipeline to support more natural text responses.
            self.model = build_local_pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,
            )
        except Exception:
            # If the model fails to load for any reason, do not crash the application.
            self.model = None

    def _clean_model_output(self, text: str) -> str:
        # Clean the raw generated text from the model to make it readable and stable.
        cleaned = text.strip()
        if not cleaned:
            return "I am ready to help with the Crypto Horizon project."

        # Remove the prompt framing that the model sometimes includes in its generated output.
        if "User:" in cleaned:
            cleaned = cleaned.split("User:", 1)[0].strip()
        if "Assistant:" in cleaned:
            cleaned = cleaned.split("Assistant:", 1)[1].strip()
        if cleaned.startswith("\n"):
            cleaned = cleaned.lstrip()

        # Keep the output concise so the UI remains readable and responsive.
        return cleaned[:800]

    def _knowledge_response(self, prompt: str) -> Optional[str]:
        # This structured rule-based layer gives reliable answers even without a loaded model.
        text = prompt.lower()

        if any(keyword in text for keyword in ["regime", "market regime", "cluster"]):
            return (
                "The project uses a 3-regime clustering model to separate market states: "
                "Balanced Stability, Market Indecision/Chop, and High-Volatility Shift. "
                "These regimes are built from BNB returns, BTC returns, and a rolling volatility signal."
            )

        if any(keyword in text for keyword in ["correlation", "pearson", "relationship"]):
            return (
                "The correlation engine measures the rolling relationship between BNB and BTC using a Pearson-style metric. "
                "This helps detect lead-lag structure and cross-asset behavioral shifts over a 14-step lookback window."
            )

        if any(keyword in text for keyword in ["lstm", "forecast", "prediction", "deep learning"]):
            return (
                "The forecasting pipeline uses an LSTM model built on BNB, BTC, and the market regime cluster. "
                "The model is trained on a 10-day sequence window, scaled with MinMax normalization, and optimized with Adam."
            )

        if any(keyword in text for keyword in ["data", "csv", "dataset", "source"]):
            return (
                "The project reads historical crypto data from Crypto_Data_Since_2015.csv and processes it with pandas. "
                "The pipeline normalizes time-series features and prepares them for clustering and forecasting."
            )

        if any(keyword in text for keyword in ["dashboard", "dash", "evaluation", "metrics"]):
            return (
                "The evaluation layer runs through a Dash dashboard and tracks model quality using loss, validation metrics, and interactive market testing."
            )

        if any(keyword in text for keyword in ["hello", "hi", "hey", "start"]):
            return "Hello. I am the Crypto Horizon local agent. I can explain the project, its market model, the correlation engine, and the LSTM forecasting workflow."
        return None

    def _fallback_response(self, prompt: str) -> str:
        # Generic fallback used when the prompt does not match a supported topic category.
        return (
            "I’m the local Crypto Horizon Agent. This project focuses on BNB/BTC market structure, regime detection, "
            "rolling correlations, and LSTM forecasting. Ask me about the dataset, the clustering model, the correlation signal, or the training pipeline."
        )

    def generate_response(self, user_prompt: str, chat_history: Optional[List[dict]] = None) -> str:
        # Validate the input before generating a response.
        if not user_prompt or not str(user_prompt).strip():
            return "Please ask a question about the Crypto Horizon project."

        # Prefer the local language model when it has been successfully loaded.
        if self.model:
            try:
                prompt_text = f"You are a helpful crypto AI analyst.\nUser: {user_prompt}\nAssistant:"
                result = self.model(
                    prompt_text,
                    max_new_tokens=120,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                )
                text = result[0]["generated_text"]
                cleaned = self._clean_model_output(text)
                if cleaned and cleaned != prompt_text:
                    return cleaned
            except Exception:
                # If generation fails, fall through to the deterministic project knowledge.
                pass

        # Product-specific knowledge layer. This keeps answers accurate for the project structure.
        knowledge_answer = self._knowledge_response(user_prompt)
        if knowledge_answer:
            return knowledge_answer

        # Final fallback response for anything not covered by the project logic.
        return self._fallback_response(user_prompt)

    def run_console(self):
        # Command-line chat loop for the project assistant.
        print("Crypto Horizon Local Agent is ready.")
        print("Type 'exit', 'quit', or 'bye' to leave.\n")

        chat_history = []
        while True:
            try:
                user_prompt = input("You: ").strip()
            except KeyboardInterrupt:
                print("\nSession ended.")
                break

            if not user_prompt:
                continue

            if user_prompt.lower() in {"exit", "quit", "bye"}:
                print("Agent: Goodbye.")
                break

            response = self.generate_response(user_prompt, chat_history)
            print(f"Agent: {response}\n")
            chat_history.extend([
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response},
            ])
            
DirectAIResponseAgent = CryptoHorizonAgent

def ask(prompt: str) -> str:
    agent = CryptoHorizonAgent()
    return agent.generate_response(prompt)

if __name__ == "__main__":
    agent = CryptoHorizonAgent()
    agent.run_console()
