import os
from openai import OpenAI

class PipelineChatbotAgent:
    def __init__(self, api_key=None):
        """
        Initializes the NLP assistant using the OpenAI API.
        Grounds the agent on the exact parameters of your 2015-2025 pipeline.
        """
        # Falls back to standard environment keys if not explicitly defined
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        
        # Grounding system parameters matching your 2015-2025 cryptocurrency pipeline cells
        self.system_context = (
            "You are an embedded financial data science assistant inside a cryptocurrency predictive dashboard.\n"
            "The engineering codebase consists of the following modular architectures that you are querying:\n\n"
            "- DATA FRAMEWORK: Managed by cell 2 via pandas. Ingests data from 'Crypto_Data_Since_2015.csv'. Converts dates, computes dynamic continuous log returns, and deletes raw artifacts from RAM for execution optimization.\n"
            "- UNSUPERVISED ML: Built on Sklearn K-Means tracking 3 distinct market regimes using BNB Returns, BTC Returns, and a 10-day moving window of BNB Volatility. Distance metrics represent an active Anomaly/AI Signal Score.\n"
            "  * Regime 0: Balanced Stability\n"
            "  * Regime 1: Market Indecision/Chop\n"
            "  * Regime 2: High-Volatility Shift\n"
            "- CORRELATION TRACKING: Cell 3 generates a standard Pearson metric using a rolling lookback size of 14 intervals.\n"
            "- DEEP LEARNING FORECASTING: Cell 5 defines a stacked TensorFlow Keras LSTM framework. Feature columns parsed: ['BNB', 'BTC', 'Market_Regime_Cluster'] transformed using a MinMaxScaler (0,1). Window sliding parameters utilize a 10-day sequence lookback size. Sequence vectors split sequentially using 80% for training data and 20% for testing data. Network layers consist of stacked dual 50-unit LSTM layers pointing to a 1-unit Dense output neuron, using an Adam optimizer over 20 epochs.\n\n"
            "Answer questions with clear, direct, and authoritative machine-learning insights. Keep explanations concise."
        )

    def generate_response(self, chat_history):
        """
        Takes the user chat message history array and returns an AI-generated completion string.
        """
        formatted_messages = [{"role": "system", "content": self.system_context}]
        
        # Append existing message loops cleanly to handle session tracking
        for message in chat_history:
            formatted_messages.append({"role": message["role"], "content": message["content"]})
            
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=formatted_messages
            )
            return completion.choices.message.content
        except Exception as e:
            return f"Pipeline Chatbot Error: Unable to complete message routing. {str(e)}"