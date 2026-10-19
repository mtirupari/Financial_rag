"""Execute notebook Step 21 against existing Pinecone namespaces (no index writes)."""
import ast
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
load_dotenv(ROOT / '.env', override=False)
for key in ('NEBIUS_API_KEY', 'PINECONE_API_KEY', 'PINECONE_INDEX_NAME'):
    if not os.environ.get(key, '').strip():
        raise RuntimeError(f'Missing {key}')

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

print('Connecting to existing Pinecone index...', flush=True)
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
description = pc.describe_index(os.environ['PINECONE_INDEX_NAME'])
if description.dimension != 768 or description.metric != 'cosine':
    raise RuntimeError('Expected a 768-dimensional cosine index.')
index = pc.Index(host=description.host)
stats = index.describe_index_stats()
for namespace in ('fixed', 'semantic'):
    count = stats['namespaces'].get(namespace, {}).get('vector_count', 0)
    print(f'{namespace}: {count} vectors', flush=True)
    if not count:
        raise RuntimeError(f'No records in {namespace}')
print('Loading cached MPNet embedding model...', flush=True)
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/multi-qa-mpnet-base-cos-v1',
    model_kwargs={'local_files_only': True, 'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32},
)
nb = json.loads((ROOT / 'paypal_sec_10k.ipynb').read_text())
cells = {cell['id']: ''.join(cell['source']) for cell in nb['cells']}
ns = dict(globals())
exec(cells['54631880'], ns)
# Import only the retrieval function, not the notebook's separate test run.
tree = ast.parse(cells['f62ab742'])
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(ast.Module(body=functions, type_ignores=[]), '<retrieval>', 'exec'), ns)
for cell_id in ('a2b0411f', 'groq-rag-chain', 'paypal-eval-dataset', 'three-experiments-collector'):
    exec(compile(cells[cell_id], cell_id, 'exec'), ns)
print('RESULT_DIRECTORY=' + str(ns['experiment_output_dir']), flush=True)
