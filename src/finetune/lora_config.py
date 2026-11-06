"""
Etapa 09 — Fine-tuning (LoRA / QLoRA / PEFT)
--------------------------------------------------
Este projeto usa RAG (contexto injetado no prompt) em vez de fine-tuning,
porque é a abordagem certa para "a base de conhecimento muda com
frequência" — que é o caso de políticas de suporte. Fine-tuning faz mais
sentido para mudar o *estilo/comportamento* do modelo (ex.: seguir um
tom de voz específico, aprender um formato de saída estruturado).

Este módulo mostra a configuração real que seria usada com a biblioteca
`peft` para um fine-tuning LoRA de um modelo open-source local (ex.:
Llama, Mistral), caso o time decida migrar de "RAG puro" para
"RAG + modelo fine-tunado" (abordagem híbrida, comum em produção).

Requer GPU — não é executado neste projeto, é o "código de referência"
documentado para quando a etapa 09 for de fato necessária.
"""
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    r: int = 16                 # rank da decomposição — mais alto = mais capacidade, mais memória
    lora_alpha: int = 32        # fator de escala (convenção comum: 2x o r)
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def build_peft_config(cfg: LoRAConfig):
    """Constrói o LoraConfig real da lib `peft`. Só importa a lib
    quando a função é chamada — evita exigir GPU/peft instalados
    para rodar o resto do projeto."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


TRAINING_NOTES = """
Fluxo de referência para fine-tuning (não executado neste projeto):

1. Preparar dataset supervisionado (pares instrução/resposta no formato
   e tom desejados) — normalmente centenas a milhares de exemplos.
2. Carregar o modelo base em 4-bit (QLoRA, via bitsandbytes) para caber
   em uma única GPU de consumo (ex.: RTX 4090 24GB para um 7B).
3. Aplicar `get_peft_model(base_model, build_peft_config(LoRAConfig()))`.
4. Treinar com o `Trainer` do Hugging Face ou TRL's `SFTTrainer`.
5. Avaliar com o MESMO conjunto de eval do RAG (eval/evaluate.py),
   comparando o modelo fine-tunado contra o baseline (etapa 10).
6. Fazer merge dos pesos LoRA no modelo base ou servir com adapters
   dinâmicos (vLLM suporta múltiplos adapters LoRA simultâneos).
"""
