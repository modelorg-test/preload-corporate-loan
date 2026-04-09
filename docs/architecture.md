# Architecture Overview

| Component | Responsibility | Tech Stack |
| --- | --- | --- |
| **Ingestion** | Streams raw metrics from Kafka queues. | `pyspark` |
| **Processor** | Maps categorical embeddings through the registry. | `scikit-learn` |
| **Inference** | Serves requests under 20ms p99 via ONNX graphs. | `fastapi` |

> [!NOTE]
> Ensure all Kafka listeners operate via explicit dead-letter-queues to prevent data poisoning.
