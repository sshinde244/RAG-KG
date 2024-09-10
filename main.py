import transformers
import torch

# Define the model ID
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize the text-generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# Define the instructional preamble
instructions = """
You are an AI language model. Please adhere to the following guidelines when responding:

1. **Be Informative**: Provide accurate and helpful information on a wide range of topics.
2. **Be Neutral and Impartial**: Avoid taking a personal stance or promoting a specific agenda, and instead strive for objectivity and fairness.
3. **Use Clear and Simple Language**: Communicate complex ideas in a way that's easy to understand, avoiding jargon and technical terms whenever possible.
4. **Respect User Privacy**: Protect users' personal information and maintain confidentiality when interacting with me.
5. **Follow Conversational Guidelines**: Engage in natural-sounding conversations, using context and understanding to respond to questions and statements.
6. **Avoid Giving Advice or Making Decisions for Users**: Provide suggestions and guidance, but ultimately let users make their own choices and decisions.
7. **Be Transparent About My Limitations**: Clearly indicate when I don't have the information or ability to answer a question or complete a task.

Additional Guidelines:
* You are trained on a massive dataset of text from various sources, including web pages, books, research papers, and conversations.
* Your training data is sourced from reputable websites and organizations, ensuring that the information you provide is accurate and trustworthy.
"""

# Initialize an empty conversation history
conversation_history = [{"role": "system", "content": instructions}]

def ask_question(prompt):
    global conversation_history
    
    # Add the user's prompt to the conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    # Define the end-of-sequence token IDs correctly
    terminators = [
        pipeline.tokenizer.eos_token_id
    ]
    
    # Generate the response based on the entire conversation history
    outputs = pipeline(
        conversation_history,
        max_new_tokens=256,
        eos_token_id=terminators[0],  # Correctly use the eos_token_id
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Extract the generated response
    generated_text = outputs[0]["generated_text"][-1]['content']
    
    # Append the model's response to the conversation history
    conversation_history.append({"role": "assistant", "content": generated_text})
    
    # Return the generated response
    return generated_text.strip()

# Define KnowledgeGraphIndex
class KnowledgeGraphIndex:
    def __init__(self):
        self.graph = {}

    def add_nodes_from_dataset(self, dataset):
        for subject, relations in dataset.items():
            self.graph[subject] = relations

    def add_relations(self, relations_text):
        for line in relations_text.split('\n'):
            if line.strip():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) == 3:
                    subject, relation, object_ = parts
                    subject, relation, object_ = subject.replace('_', ' '), relation.replace('_', ' '), object_.replace('_', ' ')
                    if subject not in self.graph:
                        self.graph[subject] = []
                    self.graph[subject].append((relation, object_))
                else:
                    pass
                    # print(f"Skipping line: {line.strip()} - does not contain exactly 3 values after stripping and splitting")

    def query(self, subject, relation):
        return [target for (rel, target) in self.graph.get(subject, []) if rel == relation]
    
# Define GraphStore
class GraphStore:
    def __init__(self):
        self.graph = {}

    def load_from_kg_index(self, kg_index):
        self.graph = kg_index.graph

    def query(self, subject, relation):
        return [target for (rel, target) in self.graph.get(subject, []) if rel == relation]

def read_dataset(file_path):
    dataset = {}
    with open(file_path, 'r') as file:
        for line in file:
            print(f"Processing line: {line.strip()}")
            parts = [part.strip() for part in line.strip().split(',')]
            if len(parts) == 3:
                subject, relation, object_ = parts
                subject, relation, object_ = subject.replace('_', ' '), relation.replace('_', ' '), object_.replace('_', ' ')
                if subject not in dataset:
                    dataset[subject] = []
                dataset[subject].append((relation, object_))
            else:
                pass
                # print(f"Skipping line: {line.strip()} - does not contain exactly 3 values after stripping and splitting")
    return dataset

def ent4search(response):
    # Find the line that starts with 'Entities:'
    for line in response.splitlines():
        if line.startswith("Entities:"):
            # Extract the entities after 'Answer: Entities:'
            entities = line.split("Entities:")[1].strip()
            # Split the entities by comma and strip any surrounding whitespace
            entity_list = [entity.strip() for entity in entities.split(',')]
            return entity_list
    return []

def search_triplets_by_entity(kg_index, entities):
    results = []
    for entity in entities:
        for relation, target in kg_index.graph.get(entity, []):
            results.append((entity, relation, target))
        for subj in kg_index.graph:
            for relation, target in kg_index.graph[subj]:
                if target == entity:
                    results.append((subj, relation, entity))
    return results

def query_user():
    prompt = input("Ask a question: ")
    if prompt.strip():
        answer = ask_question(prompt)
        print(f"Answer: {answer}")

def main(dataset_file):

    # STEP 1
    print("Reading dataset...")
    your_dataset = read_dataset(dataset_file)
    print("Dataset loaded successfully.")        

    # STEP 2
    print("Initializing KnowledgeGraphIndex...")
    kg_index = KnowledgeGraphIndex()
    kg_index.add_nodes_from_dataset(your_dataset)
    print("KnowledgeGraphIndex populated.")

    # STEP 3
    print("Loading GraphStore...")
    graph_store = GraphStore()
    graph_store.load_from_kg_index(kg_index)
    print("GraphStore loaded.")

    # STEP 4
    # Load LLM with instructions and machining query
    prompt = 'What were the exact instructions given to you?'
    intro = ask_question(prompt)
    print(intro)

    # STEP 5
    # Ask user for machining query, print LLM response, convert response to triples
    query_user()
    prompt = '''
    First, save the previous question to memory as "User_Query". Now, take your response to "User_Query" and convert it from natural language to triplets.
    This should be done by extracting information relevant to the query from your answer
    and then formatting it as [subject, relation, object] triplets, such as: part_A, made_from, steel.
    Like in the example above, replace spaces between words with underscores. 
    '''
    ans2ent = ask_question(prompt)
    print(ans2ent)

    # STEP 6
    # Read LLM's response with triples, add to KG, refresh KG
    kg_index.add_relations(ans2ent)
    graph_store.load_from_kg_index(kg_index)
    print(f'Knowledge graph updated.')

    # STEP 7
    # Extract entities from user's query, prepare to search KG for entities
    prompt = '''
    Extract all entities from "User_Query". Give them back to me as a comma separated list starting with "Entitites:".
    For example, if "User_Query" was "What drill is the cylindrical hole made by?", some entities you could extract might include "drill, cylindrical_hole, made_by", etc.
    Note how spaces between words were replaced by underscores for applicable entities, and how entities resemble either nouns or verbs in the query.
    The words "subject", "relation", or "object" should not be explicitly included in your list of entities.
    '''
    response =  ask_question(prompt)
    entities = ent4search(response)
    print(f"Extracted Entities List: {entities}")
    
    # # STEP 8
    # results = search_triplets_by_entity(kg_index, entities)
    # print(f"Search results from KG: {results}")
    


if __name__ == "__main__":
    dataset_file = r"triplets.txt"
    main(dataset_file)
        
