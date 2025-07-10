# Poly Scholar

Poly Scholar is a multi-agent AI system designed to facilitate literature search, summarization, and gap analysis in academic research. Built on the LangGraph framework, this project leverages various tools and agents to streamline the research process.

## Overview

The Poly Scholar project consists of several components that work together to provide a comprehensive solution for academic inquiries. The system includes agents for literature searching, summarization, and identifying gaps in research, as well as tools for web searching and metadata retrieval from arXiv.

## Features

- **Multi-Agent System**: Coordinated agents that handle different aspects of the research process.
- **Efficient Retrieval**: Utilizes vector indexing and embeddings for fast information retrieval.
- **Dynamic Prompt Management**: Centralized management of prompt templates for various tasks.
- **Memory Management**: Implements both short-term and long-term memory for enhanced context retention.

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone https://github.com/yourusername/poly_scholar.git
   cd poly_scholar
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8 or higher installed. Then, install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Configuration**:
   Modify the `config/config.yaml` file to set your desired parameters.

4. **Run the Application**:
   Start the application using the following command:
   ```
   python src/deployment/server.py
   ```

## Usage Guidelines

- To initiate a literature search, use the Supervisor agent to coordinate tasks among other agents.
- Utilize the provided tools for web searches and metadata retrieval as needed.
- Refer to the documentation in the `docs/` directory for detailed architecture and API references.

## Testing

Unit tests are included in the `tests/` directory. To run the tests, execute:
```
pytest tests/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.