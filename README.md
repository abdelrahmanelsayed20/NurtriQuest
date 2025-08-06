﻿# NutriQuest 💪

A sophisticated nutrition and workout information retrieval system that helps users find personalized diet and exercise plans.

## About NutriQuest

NutriQuest is an advanced information retrieval application that uses natural language processing (NLP) and machine learning techniques to provide users with relevant nutrition and workout information. The application allows users to search for diet plans, workout routines, and health advice from multiple sources including fitness websites, nutrition databases, and research articles.

## Features

- **Intelligent Search**: Utilizes BM25 ranking algorithm and TF-IDF for accurate information retrieval
- **Query Expansion**: Automatically enhances user queries with related terms for better search results
- **Multi-source Search**: Filters results by specific information sources
- **NLP Processing**: Implements tokenization, stemming, and stopword removal for improved search precision
- **Performance Metrics**: Calculates precision, recall, F1 score, and NDCG to evaluate search quality
- **User-friendly Interface**: Clean, intuitive Streamlit web application interface
- **AI-powered Chatbot**: Interactive assistant that provides personalized nutrition and fitness advice

## Live Demo

The application is hosted and available online at:
https://nutriquest.streamlit.app/

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/XMostafaNashaatX/Nurtri_Quest.git
   cd Nurtri_Quest
   ```

2. Create a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Download NLTK resources (automatically handled on first run)

## Usage

1. Run the Streamlit application:

   ```
   streamlit run "app (1).py"
   ```

2. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

3. Enter your nutrition or workout-related queries in the search box

4. Use the available filters to refine your search results

5. Toggle the "AI Enhanced Chatbot" option in the sidebar to switch to the conversational assistant

## AI Assistant

NutriQuest features an AI-powered chatbot that provides personalized nutrition and fitness guidance:

- **Knowledge-Based Responses**: The chatbot utilizes the NutriQuest document database to provide evidence-based answers
- **Contextual Understanding**: Maintains conversation history to provide relevant follow-up responses
- **Customizable Models**: Administrators can select different AI models for varied capabilities (powered by OpenRouter API)
- **Multi-Source Integration**: Pulls information from Wikipedia, YouTube, and Google search results
- **User-Friendly Interface**: Clean chat interface with easy-to-read message bubbles and smooth animations

## Technology Stack

- **Streamlit**: Web application framework
- **NLTK**: Natural Language Processing toolkit for text preprocessing
- **PyTerrier**: Information retrieval system for document indexing and ranking
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Requests**: HTTP requests for external API integration
- **OpenRouter API**: AI model integration for the chatbot functionality

## Project Structure

The main components of NutriQuest include:

- Text preprocessing and cleaning functions
- Document indexing and retrieval mechanisms
- Multiple ranking models (BM25, TF-IDF)
- Query expansion using RM3 technique
- Performance evaluation metrics
- User interface with customization options
- AI chatbot integration with OpenRouter

## Contributing

Contributions to NutriQuest are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or suggestions, please open an issue on GitHub or contact the project maintainer at s-mostafa.abdelhameed@zewailcity.edu.eg.  or at S-abdulrahman.hassan@zewailcity.com

