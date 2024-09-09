import sys
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the CompGCN directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'compgcn')))

# Initialize Llama (LLM)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Initialize the text-generation pipeline
model = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=0,  # Use GPU (assuming you have a compatible CUDA device)
)

# Define stop tokens
stop_tokens = ["<|endoftext|>", "\n"]

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

# Other functions
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

def load_llm(intro):
    outputs = model(intro, max_length=500, num_return_sequences=1, truncation=False)
    generated_text = outputs[0]['generated_text']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]
            break
    return generated_text

def triplets_from_llm_response(query):
    input_text = f'''Answer this question: {query}. 
    Then, take your response to the question and convert it from natural language to triplets.
    This should be done by extracting information relevant to the query from your response
    and then formatting it as [subject, relation, object] triplets. 
    Omit the brackets when actually formatting and returning them. 
    For example, if the the question I give you as input is "Describe steel and its relationship with part_a", and your answer was:
    "Steel is a type of metal and a material commonly used for machining. For example, Part_A is made from steel."
    then the triplets you could extract from your response would be:
    steel, is_a, material
    steel, type, metal
    part, made_from, steel
    When you print your response, print for me both the natural language response and the triplets.
    When you have completed this task, stop generating your response.
    '''
    outputs = model(input_text, max_length=500, num_return_sequences=1, truncation=False)
    generated_text = outputs[0]['generated_text']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]
            break
    return generated_text.replace('_', ' ')

def entities_from_query(query):
    input_text = f'''Extract any entities (ex. subject, object, relations, etc.) from this query:{query}.
    Return any extracted items in a comma separated list, for example:
    steel_mill, drill, part_A
    When you have completed this task, stop generating your response.
    '''
    outputs = model(input_text, max_length=500, num_return_sequences=1, truncation=False)
    generated_text = outputs[0]['generated_text']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]
            break
    return generated_text.replace('_', ' ')

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

def save_triplets_to_dict(triplets):
    triplets_dict = {}
    for triplet in triplets:
        subject, relation, object_ = triplet
        if subject not in triplets_dict:
            triplets_dict[subject] = []
        triplets_dict[subject].append((relation, object_))
    return triplets_dict
            
def extract_entities_from_query(query):
    input_text = f'''Extract all entities from the following query: {query}. In your response, do not include or say anything other than the list.
    Your response should look like: 'steel_mill, drill, hole_diameter". No other words,
    and replace spaces between words with underscores.
    When you have completed this task, stop generating your response.
    '''
    outputs = model(input_text, max_length=500, num_return_sequences=1, truncation=False)
    generated_text = outputs[0]['generated_text']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]
            break
    entities = [entity.strip() for entity in generated_text.split(',')]
    return entities
    
def convert_triples_to_answer(triplets_dict, query):
    input_text = f'''Read this entire dictionary: {triplets_dict}. Each key in the dictionary represents a subject entity. 
    The two values for each key are the relation and object entity, respectively. Together, the form a triplet in the form of 
    [subject, relation, object], for example: steel_mill, drills, part_A
    Use the triples in this dictionary in order to formulate an answer this question: {query}.
    Only use the information provided, do not generate further information.
    When answering, provide only the answer and no unnecessary text. For example:
    'Steel mills are used for creating steel parts.'
    Only include relevant information in your answer. Sometimes, the dictionary will have more information
    than needed, and you do not have to use it all, but your answer should be relevant and thorough.
    When you have completed this task, stop generating your response.
    '''
    outputs = model(input_text, max_length=500, num_return_sequences=1, truncation=False)
    generated_text = outputs[0]['generated_text']
    for stop_token in stop_tokens:
        if stop_token in generated_text:
            generated_text = generated_text.split(stop_token)[0]
            break
    return generated_text.replace('_', ' ')
    
# Main function to integrate all parts
def main(dataset_file, output_file):
    try:
        # STEP 1: INITIALIZE LOCAL KG
        print("Reading dataset...")
        your_dataset = read_dataset(dataset_file)
        print("Dataset loaded successfully.")
        
        print("Initializing KnowledgeGraphIndex...")
        kg_index = KnowledgeGraphIndex()
        kg_index.add_nodes_from_dataset(your_dataset)
        print("KnowledgeGraphIndex populated.")

        # STEP 2: STORE LOCAL KG TO MEMORY FOR FUTURE ACCESS
        print("Loading GraphStore...")
        graph_store = GraphStore()
        graph_store.load_from_kg_index(kg_index)
        print("GraphStore loaded.")

        # STEP 3: PROMPT LLM WITH QUERY (enter prompt here), extract triplets from LLM response
        intro = '''
        Your purpose is to answer questions using knowledge from CNC data that will be given to you along with your own intelligence. 
        Follow the directions exactly as they are given to you.
        Save these instructions to memory.
        '''
        load_llm(intro)

        query = "What causes calibration errors?"
        print(f"Processing query: {query}")
        generated_knowledge = triplets_from_llm_response(query)
        print(f"Triplets to be added to knowledge graph: {generated_knowledge}")

        # STEP 4: ADD EXTRACTED TRIPLETS FROM LLM RESPONSE TO KG
        kg_index.add_relations(generated_knowledge)

        # STEP 5: RELOAD KG WITH UPDATED TRIPLETS
        graph_store.load_from_kg_index(kg_index)
        print(f'Knowledge graph updated.')

        # STEP 6: EXTRACT ENTITIES FROM PROMPT, SEARCH KG FOR ENTITIES
        entities = extract_entities_from_query(query)
        print(f'Entities extracted from query: {entities}.')
        print(f'Searching Knowledge Graph...')
        results = search_triplets_by_entity(kg_index, entities)
        print(f"Search results for entities: {results}")

        # STEP 7: Save the results to a dictionary
        triplets_dict = save_triplets_to_dict(results)
        print(f"Triplets to be given to LLM: {triplets_dict}")
        
        # STEP 8: Give dictionary to LLM for reasoning and question answering
        answer = convert_triples_to_answer(triplets_dict, query)
        print(f"The answer to your query: {answer}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    dataset_file = r"example_dataset.txt"
    output_file = 'triplets_from_KG_search.txt'
    main(dataset_file, output_file)
