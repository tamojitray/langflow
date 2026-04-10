from .model_metadata import create_model_metadata

# Unified model metadata for Azure OpenAI
AZURE_OPENAI_MODELS_DETAILED = [
    create_model_metadata(provider="Azure OpenAI", name="gpt-4o-mini", icon="Azure", tool_calling=True),
    create_model_metadata(provider="Azure OpenAI", name="gpt-4o", icon="Azure", tool_calling=True),
    create_model_metadata(provider="Azure OpenAI", name="gpt-4-turbo", icon="Azure", tool_calling=True),
    create_model_metadata(provider="Azure OpenAI", name="gpt-4", icon="Azure", tool_calling=True),
    create_model_metadata(provider="Azure OpenAI", name="gpt-3.5-turbo", icon="Azure", tool_calling=True),
    # Reasoning Models
    create_model_metadata(provider="Azure OpenAI", name="o1", icon="Azure", reasoning=True),
    create_model_metadata(provider="Azure OpenAI", name="o1-mini", icon="Azure", reasoning=True),
    create_model_metadata(provider="Azure OpenAI", name="o3-mini", icon="Azure", reasoning=True, not_supported=True),
]

AZURE_OPENAI_EMBEDDING_MODELS_DETAILED = [
    create_model_metadata(
        provider="Azure OpenAI",
        name="text-embedding-3-small",
        icon="Azure",
        model_type="embeddings",
    ),
    create_model_metadata(
        provider="Azure OpenAI",
        name="text-embedding-3-large",
        icon="Azure",
        model_type="embeddings",
    ),
    create_model_metadata(
        provider="Azure OpenAI",
        name="text-embedding-ada-002",
        icon="Azure",
        model_type="embeddings",
    ),
]
