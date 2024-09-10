Retrieval Augmented Generation model using local KG and Llama3
* made for Intelligent Systems Control Laboratory, UConn Department of Mechanical Engineering

Note: this query answering system uses the recently deployed Meta Llama3 8B Instruct model, readily available in the HuggingFace library. The LLM is used throughout this program for text generation and question answering. For this code to work, the model must be ran on GPU.

The Llama3 model is intialized with predetermined instructions and a conversation storage feature is implemented in order to ensure long-term memory for future conversations with the model. The knowledge graph (KG) is initialized and stored by KnowledgeGraphIndex for future use and edits. 
The dataset stored is comprised of thousands of relational triplets in the form of [subject, relation, object], which were all collected and processed by myself from several research papers and publications on manufacturing techniques, but specifically CNC machining. 

This program focuses specifically on implementing a query answering system for a user who can prompt it with questions related to CNC machining, milling, and tools. However, this program can work for any dataset by simply changing the input knowledge graph and making slight edits to the prompts given to the LLM. While more widely used AI can also produce answers to related questions, my program incororates parallel thinking by working in the knowledge graph into the LLM's answer reasoning processes with the aim of reducing AI hallucination and errors. 

This is done by having the LLM answer the desired user query, and then using natural language processing in order to convert the LLM's response into knowledge triplets in the same [subject, relation, object] form. These triplets, once extracted and converted from the response, are then added to the knowledge graph. The updated KG is now populated with additional information from the LLM's response. 

More processing is conducted on the query, but now to extract entities of the same manner as the triplets. Once completed, several functions are run in order to search the updated KG for triplets containing or related to the extracted entities, which now include knowledge from the LLM. This subset of triplets is then given back to the LLM for analysis, which determines which relational triplets are most relavant in order to formulate it's final answer. The answer is then returned to the user.
