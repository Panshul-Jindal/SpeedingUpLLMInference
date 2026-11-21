# Background 

Serving a large language model (Inference)  is dominated by system constraints rather than model quality. LLM Inference can be divided into two parts: the prefilling and decoding stages. Decoding latency is largely bounded by memory bandwidth, as decoding is an autoregressive process with very low AI. (As the amount of computation happens for one new token generation, \<\<  the amount of memory that should be supplied) . Because memory bandwidth dominates decoding, model size is critical to performance. Larger models require transferring more parameters per token, directly slowing down generation latency. In contrast, Prefilling processes prompt tokens in parallel and is a compute-bound process with high arithmetic intensity (AI).   
KV Caching is widely used for inference because it reduces the need to recompute keys and values for each new query. The amount of memory that has to be supplied for a batch:

***B of requests theoretically  \= Model weights \+ B\*KV cache***

High-bandwidth memory (HBM) and SRAM limits constrain the maximum batch size B. So, B cannot be too huge, and B is what determines the throughput; thus, the achievable throughput is bounded by how much KV cache fits in memory and how effectively requests are batched ( we do not want long requests to stall many small requests). Though many strategies exist to reduce the KV Cache size itself in the first place, like MQA, GQA (without affecting accuracy much).

However, the gap is closed mainly through infrastructure-level techniques:

* **Weight & KV Cache Quantization:** Using reduced-precision representation (e.g., INT8/FP8 or INT4) to decrease memory footprint and increase throughput.  
* **Context Reuse:** Reusing previously computed KV caches for shared prompt prefixes (e.g., RadixAttention / SGLang).  
* **Advanced Scheduling & Parallelism:** Employing chunked prefilling (e.g., Sarathi-Serve) or optimal tensor/pipeline parallelism to mitigate compute-memory imbalances.  
* **Multi-Token Generation:** Using speculative decoding to draft and verify multiple tokens per forward pass.

These techniques are proposed separately, and many are specific to a particular kind of end task, such as KVCache Blending for a RAG-based workflow. Thus, these techniques are not independently composable in practice.   
Take, for example, reduced precision and speculative decoding:

* Reducing precision frees memory and changes the optimal concurrency, which changes whether a model should be sharded or replicated;   
* Speculative decoding converts idle compute into tokens at low load but competes with useful work at high load. Applying them without regard to interaction can yield no net improvement. Timing is the problem; i.e., it should be applied when there is low load


 Also, most techniques deliver gains only in specific regimes. 

* **Prompt Structure:** Prefix caching (e.g., Radix trees) yields significant gains in multi-turn chatbots or RAG workflows, but adds unnecessary tracking overhead to single-turn requests with unique prompts.  
* **Context Length & Concurrency:** Chunked prefilling (e.g., Sarathi-Serve) reduces prefill-induced tail latency under high concurrency, but provides minimal benefit under low-concurrency or short-context workloads. Iteration-level scheduling requires continuous streaming arrivals to outperform traditional batching.

Each determines whether a given optimization pays off — so an approach effective in one deployment may be neutral or harmful in another.

# Problem Statement

***The goal of this project is to substantially improve the latency and throughput of a served open-weight model by applying and combining infrastructure-level optimizations, and to establish which of them contribute the gains, under which operating conditions, and where they cease to be effective.***

To limit the scope, optimizations are classified as:

* Single Instance Optimizations:  
  * Quantized weights  
  * Reduced-precision KV cache  
  * Speculative decoding to produce multiple tokens per model pass  
* Workload-dependent optimizations:  
  * Reuse of cached context across requests sharing common prefixes  
  * Tuned batching (Choosing the appropriate batch size to ensure full compute utilization)   
  * Prefill scheduling   
  * Appropriate parallelism strategy in the prefill and decode phases

Analysis would be done across these regimes:

* Context length  
* Concurrency  
* Prompt-reuse structure, i.e ratio of requests with prompt reuse 

Key Challenges

* Optimizations can counteract each other. For example, quantization reduces weight transfer time, which narrows the idle-compute window that speculative decoding relies on to be effective.  
* Techniques beneficial in one regime may become bottlenecks under another regime (e.g., speculative decoding under low load may bottleneck in high load by stealing compute resources from batch processing).  
* Optimizations must be continually validated to ensure output quality does not degrade relative to baseline serving.

# Expected Output

There are 7 optimizations; thus, the number of optimizations to evaluate across \= 127   
Each regime can be represented by low, mid, or high  for the sake of convenience 

| Optimizations\\Regimes  | CL (Low) | CL (Mid) | CL(High) | CR (Low) | CR (Mid) | CR (High) | PR (Low) | PR (Mid) | PR (High) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Default |  |  |  |  |  |  |  |  |  |
| Conf 1 |  |  |  |  |  |  |  |  |  |
| … |  |  |  |  |  |  |  |  |  |
| Conf 127  |  |  |  |  |  |  |  |  |  |
| Final Configuration |  |  |  |  |  |  |  |  |  |

***Value \= f(Measured Latency \- Default Latency, Measured Throughput \- DefaultThroughput )***

##### CL \= Context Length, CR \= aConcurrency, PR \= Prompt-Reuse Structure. Default Latency and Default Throughput correspond to a naive implementation without any optimizations applied.

We need to identify the binding constraint in the final configuration, in which the chosen configuration is and is not the right one. Also, analysis of Techniques found to give no benefit in a given regime, with an explanation of why, is a valid part of this outcome.

