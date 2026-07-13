# Etapa 18 — Vector DBs em Escala

Este projeto usa `src/vector_store.py` (in-memory, numpy) para rodar
offline e sem dependências externas. Em produção, a base de conhecimento
cresce além do que cabe em memória de um processo só, e aí entra um
vector DB dedicado.

## Quando migrar
- Base de conhecimento > ~100k chunks, ou
- Necessidade de múltiplos workers/réplicas compartilhando o mesmo índice, ou
- Necessidade de filtros por metadata em escala (ex.: buscar só documentos
  de um cliente específico) com baixa latência.

## Opções e quando usar cada uma

| Vector DB | Quando faz sentido |
|---|---|
| **Pinecone** | Totalmente gerenciado, menor esforço operacional, bom para times pequenos |
| **Weaviate** | Open-source, suporte forte a busca híbrida (vetor + BM25) |
| **Milvus** | Open-source, melhor para volumes muito grandes (bilhões de vetores) |
| **pgvector** | Se você já usa Postgres — menos um sistema para operar |

## Troca de implementação

A interface do `InMemoryVectorStore` (`add`, `search`) é o contrato que
qualquer vector DB real implementaria. Exemplo de adaptação para Pinecone:

```python
# src/vector_store_pinecone.py
import pinecone

class PineconeVectorStore:
    def __init__(self, index_name: str):
        self.index = pinecone.Index(index_name)

    def add(self, documents, vectors):
        self.index.upsert(vectors=[
            (doc.doc_id, vec.tolist(), doc.metadata)
            for doc, vec in zip(documents, vectors)
        ])

    def search(self, query_vector, k=3):
        result = self.index.query(vector=query_vector.tolist(), top_k=k, include_metadata=True)
        return [...]  # mapear para SearchResult
```

O resto do pipeline (`rag_pipeline.py`) não muda — ele só depende da
interface `search(query_vector, k) -> list[SearchResult]`.

## Também seria necessário trocar o embedder

O TF-IDF local (`src/embeddings.py`) é léxico, não semântico — funciona
para este projeto de demonstração mas não captura paráfrases sem overlap
de palavras. Em produção, usar um embedder denso real:
`text-embedding-3-small` (OpenAI), `embed-v3` (Cohere), ou um modelo
local via `sentence-transformers`.
