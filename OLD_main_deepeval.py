import os
import glob
from dotenv import load_dotenv

# --- NEW IMPORTS FOR BEDROCK/LITELLM SUPPORT ---
from deepeval.models.base_model import DeepEvalBaseLLM
from litellm import completion
# -----------------------------------------------

from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import (
    StylingConfig, 
    EvolutionConfig, 
    ContextConstructionConfig
)

# 1. Load Environment Variables
load_dotenv()

# --- NEW: DEFINE AWS CREDENTIALS (if not in .env) ---
# Make sure these are in your .env or set here
# os.environ["AWS_ACCESS_KEY_ID"] = "your_key"
# os.environ["AWS_SECRET_ACCESS_KEY"] = "your_secret"
# os.environ["AWS_REGION_NAME"] = "us-east-1"
# ----------------------------------------------------

# Check for API Key (Keeping this as requested, DeepEval may still use it for embeddings)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file")

# --- NEW: CUSTOM WRAPPER CLASS FOR BEDROCK ---
class BedrockWrapper(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        # This sends the prompt to AWS Bedrock via LiteLLM
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # Async version required by DeepEval
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def get_model_name(self):
        return self.model_name
# ---------------------------------------------

def generate_chilean_bank_testset():
    print("--- Starting Synthetic Data Generation PoC ---")

    # 2. Define File Paths
    # Assuming files are in a folder named 'knowledge_base'
    kb_folder = "../old_knowledge_base_small"
    document_paths = glob.glob(os.path.join(kb_folder, "*.md"))
    
    if not document_paths:
        raise FileNotFoundError(f"No .md files found in {kb_folder}. Please add your 5 files.")
    
    print(f"Found {len(document_paths)} documents.")

    # --- DEBUG STEP: Verify files are readable ---
    # This prevents the "0 out of 0 chunks" error by catching encoding issues early.
    print("Verifying file readability...")
    for path in document_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    print(f"[WARNING] File is empty: {path}")
        except Exception as e:
            print(f"[ERROR] Could not read {path}. Rename the file to remove special characters (accents). Error: {e}")
            return # Stop execution if files are unreadable
    # ---------------------------------------------

    # 3. Configure Styling (Language & Currency Context)
    styling_config = StylingConfig(
        input_format="Preguntas en español. Tomar el rol de un usuario promedio chileno. Las preguntas deben ser realistas. Un usuario pregunta cosas generales, no específicas. El usuario no tiene conocimiento técnico. El usuario no conoce qué se requiere, por lo que no menciona lo que tiene actualmente. Sus preguntas son conversacionales e informales, no tan correctas",
        expected_output_format="Respuestas claras y útiles en español. NUNCA se debe responder más de lo que se encuentra en el documento. Las respuestas SÓLO pueden incluir el contenido del documento.",
        task="Asistente virtual para un banco chileno especializado en créditos hipotecarios y educación financiera.",
        scenario="Un cliente chileno está haciendo preguntas sobre productos bancarios, tasas, o consejos financieros. Las preguntas las responde el documento, pero no preguntan específicamente por detalles del documento. EL USUARIO NO HA LEÍDO EL DOCUMENTO, POR LO QUE NO PUEDE PREGUNTAR POR DETALLES TÉCNICOS O ESPECÍFICOS DEL DOCUMENTO.",
    )

    # 4. Configure Evolution (RAG-Safe)
    # We avoid 'REASONING' or 'HYPOTHETICAL' to prevent hallucinations outside the KB.
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.MULTICONTEXT: 0.25,
            Evolution.CONCRETIZING: 0.25,
            Evolution.CONSTRAINED: 0.25,
            Evolution.COMPARATIVE: 0.25
        },
        num_evolutions=1  # Default complexity
    )

    # 5. Configure Context Construction (NO CHUNKING)
    # We use a large chunk_size (5000) to ensure the whole file is treated as 1 chunk.
    # encoding='utf-8' ensures we don't crash on Spanish accents.
    context_construction_config = ContextConstructionConfig(
        max_contexts_per_document=1,
        chunk_size=5000, # Large value prevents splitting the file
        chunk_overlap=0,
        encoding="utf-8"
    )

    # 6. Initialize Synthesizer
    # --- CHANGED: Use the custom BedrockWrapper instead of a string ---
    # You can change the model ID below to any Bedrock model (e.g., meta.llama3-70b-instruct-v1:0)
    # bedrock_model = BedrockWrapper(model_name="us.anthropic.claude-3-5-sonnet-20240620-v1:0")
    # bedrock_model = BedrockWrapper(model_name="openai.gpt-oss-120b-1:0")
    bedrock_model = BedrockWrapper(model_name="us.meta.llama4-maverick-17b-instruct-v1:0")

    
    synthesizer = Synthesizer(
        model=bedrock_model, 
        styling_config=styling_config,
        evolution_config=evolution_config
    )
    # ---------------------------------------------------------------

    # 7. Generate Goldens
    print(f"Generating goldens from {len(document_paths)} documents...")
    
    generated_goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        context_construction_config=context_construction_config,
        include_expected_output=True,
        max_goldens_per_context=1
    )

    # Safety check before saving
    if not generated_goldens:
        print("[ERROR] No goldens were generated. Please check the logs above for '0 out of 0 chunks'.")
        return

    # 8. Save Output
    output_filename = "chilean_bank_goldens"
    print(f"Generation complete. Saving to {output_filename}.json and .csv...")
    
    # Save as JSON (Standard DeepEval format)
    synthesizer.save_as(
        file_type='json',
        directory="./synthetic_data",
        file_name=output_filename
    )
    
    # Save as CSV (Easier for non-technical stakeholders to review in Excel)
    synthesizer.save_as(
        file_type='csv',
        directory="./synthetic_data",
        file_name=output_filename
    )

    # Optional: Convert to Pandas to show a preview
    df = synthesizer.to_pandas()
    print("\nPreview of Generated Data:")
    print(df[['input', 'expected_output']].head())

if __name__ == "__main__":
    generate_chilean_bank_testset()