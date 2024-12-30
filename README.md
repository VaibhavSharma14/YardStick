# YardStick

# RAG-Based QA Bot

## Overview
This project implements a Retrieval Augmented Generation (RAG) model for a Question-Answering (QA) bot using the OpenAI API and Pinecone as a vector database. The bot is designed to provide accurate answers to user queries based on the information contained in uploaded documents.

## Features
- Upload text documents for processing.
- Retrieve relevant information using vector embeddings.
- Generate responses using OpenAI's language model.
- User-friendly interface built with Streamlit.

## Technologies Used
- **Python**: Programming language used for development.
- **OpenAI API**: For generating human-like responses.
- **Pinecone**: Vector database for efficient similarity search.
- **LangChain**: Framework for building applications with LLMs (Large Language Models).
- **Streamlit**: For creating the web application interface.

## Installation

1. Clone the repository:

2. Install the required packages:


3. Set up your API keys:
- Create a `.env` file in the root directory and add your OpenAI and Pinecone API keys:
  ```
  OPENAI_API_KEY=your_openai_api_key
  PINECONE_API_KEY=your_pinecone_api_key
  ```

## Usage

1. Run the application:

2. Open your web browser and go to `http://localhost:8501`.

3. Upload a text document using the sidebar, then enter your query in the input box and click "Get Answer" to receive a response.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [OpenAI](https://openai.com/) for providing the language model.
- [Pinecone](https://pinecone.io/) for offering a powerful vector database solution.
- [Streamlit](https://streamlit.io/) for enabling easy web app development.

