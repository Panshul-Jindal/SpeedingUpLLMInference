# Presentation Script: Combining LLM Serving Optimizations (Final Report)

Project 24, Group 2 · Panshul Jindal, Mohd Shozab · 14 slides, about 17 minutes plus questions.
The same text is in each slide's speaker notes.

## Slide 1: Combining LLM Serving Optimizations (title)

(~45 s) Hello. This is our final report on combining infrastructure-level LLM serving optimizations. The question: when you stack optimizations like FP8 quantization, speculative decoding and prefix caching, which ones actually produce the gain, in which workloads, and where do they stop helping? The circles are the seven optimizations from the problem statement: amber are single-instance, teal are workload-dependent. The two dashed ones, chunked prefill and parallelism, are implemented in our harness but were not measured; I'll explain why on the next slide.

## Slide 2: From Plan to Study

(~1.5 min) What changed from the plan in Assignment 2. The plan was all seven optimizations, 512 cells. We ran five. The reason is computability. We decided it on 2 November, while we still had a 4-GPU node, and kept it when that node was downgraded to one with two newer H100 GPUs on 5 November. Parallelism needs more than one GPU per cell; in our harness every layout, including the default tensor-parallel one, must use all of a cell's GPUs. So with parallelism in the design, every cell occupies the whole 2-GPU node and we lose running two cells side by side. And two extra factors means four times the cells. At the measured 10.8 GPU-minutes per cell, that is roughly 90 hours of node time instead of the 11.5 we actually used, eight times longer, close to our whole budget. Chunked prefill mostly protects tail latency rather than throughput, which is our metric. Both stay implemented and are exercised in a GPU-free dry run. The other changes: we got H100s, so true FP8 instead of the W8A16 workaround, and unprivileged telemetry instead of DCGM. The time we saved paid for a second phase, which turned out to matter a lot.

## Slide 3: Study Setup

(~1 min) The setup. Five switches: FP8 weights, FP8 KV cache, EAGLE-3 speculative decoding, prefix caching, and a tuned batch limit. Seven workload points: a centre with 2048-token prompts, 32 requests in flight and 50% prompt reuse, then one regime pushed low or high at a time. Full factorial at the centre, half-fractions elsewhere: 384 runs in phase 1 plus 126 in phase 2, and 18 profiler traces. The harness pipeline is the one from the progress report. Default really has everything off, and Value only counts if both p99 latency gates and the accuracy gate pass.

## Slide 4: Best Configuration per Workload

(~1.5 min) The headline. Under the original accuracy rule, which we call policy A (the trimmed first evaluation with a one-point gate), the best configurations reach 1.39× to 2.62× the Default throughput, and every one of them meets both latency gates. Speculative decoding plus prefix caching wins at six of the seven workloads; tuned batching makes no difference, so the 90% reuse winner is the same thing. The exception is long, 8k-token prompts, where EAGLE-3 adds little and prefix plus batch wins narrowly, by under 1% over spec plus prefix. Biggest gains: 2.62× at 90% prefix reuse and 2.48× when the GPU is under-used at 2 requests in flight. Smallest: 1.39× with no prompt reuse. Note that no configuration failed a latency gate; everything that was excluded was excluded for accuracy, which I'll come back to.

## Slide 5: Critical Failure: A Hidden Co-tenant on GPU 0

(~2 min) This is the most important methodological result. The cluster node does not isolate GPUs. When we re-ran phase-1 cells in phase 2, every run on GPU 0 after 11:24 UTC on 7 November came out at about 0.6 of its true throughput. GPU 1 was fine. NVML showed 4.3 GB of extra device memory, a higher clock and lower power on GPU 0: the signature of another user's process. 71 of the 384 phase-1 runs were affected. And repetitions inside a contaminated cell still looked ordinary, CV 0.2 to 3.8%: small next to a 40% slowdown, though about ten times the spread of clean cells. We were not checking for that, so only a re-run in a different session revealed it. Because the slowdown varied between 0.56 and 0.69, we did not rescale; we replaced 8 cells with clean re-runs, excluded 16, and added 16 that phase 2 measured. The chart shows the effect on the headline: 4.5× at 90% reuse was really 2.62×, and 2.64× at high concurrency was really 1.67×. Lesson: re-run a sample of cells; tight repetitions are not proof of a clean measurement.

## Slide 6: Which Optimization Matters Where

(~1.5 min) Main effects: the average multiplier of each optimization over all the others, fitted per workload; the fit explains over 99.7% of the variance everywhere. FP8 weights are the steadiest lever, 1.22 to 1.34× at every point. EAGLE-3 is the biggest lever when the GPU has spare capacity: 2.31× at two requests in flight, but on average nothing at 8k-token prompts. Prefix caching follows reuse: nothing at 0% reuse, but also no measurable overhead, and 1.70× at 90%. FP8 KV cache only helps where the KV cache is large, 1.13× at long prompts. Tuned batching never helps, because the engine's default limit of 256 sequences is above every in-flight count we tested, so the limit never binds. Every one of these 11 no-benefit cases has a reason from telemetry.

## Slide 7: Combinations Multiply, and Speculation Compounds

(~1.5 min) Do optimizations add, multiply or cancel? For the 50 combinations where every single-optimization speedup was measured, we compared the observed speedup with the product of the single speedups. A ratio of 1 means they multiply exactly. None of the 50 fell more than 5% below the product, so in the combinations we could test, at four of the seven workloads, optimizations never cancelled each other. 32 match the product. 18 beat it by more than 5%, and every one of those contains speculative decoding; the largest is plus 31%, FP8 weights plus spec plus prefix plus batching at 90% reuse. Practically: predict combinations by multiplying, not adding; at the centre that has half the error. One synergy: FP8 weights give 1.20× alone but 1.34× on top of EAGLE-3. Our interpretation: with speculation each engine step processes 928 instead of 284 tokens, a bigger matrix multiply, which is where FP8 should help most.

## Slide 8: Prefix Caching Is Worth More with Speculation

(~1 min) The strongest synergy. We swept prompt reuse with and without speculative decoding. Alone, prefix caching gives 1.52× at 90% reuse; on top of EAGLE-3 it gives 1.85×. The interaction is flat up to 25% reuse and appears at 50%. The telemetry is consistent with a mechanism: at 90% reuse prefix caching cuts each request's decode phase by 27% without spec and 45% with it. Our interpretation: vLLM mixes prompt processing and decoding in the same engine steps; speculation shrinks the decode share, so the prompt work that prefix caching removes is a larger part of what's left. We haven't pinned down the exact threshold.

## Slide 9: Where It Stops: EAGLE-3 Beyond ~4k Tokens

(~1.5 min) Where an optimization stops helping. EAGLE-3 gives 1.35× at 2k-token prompts, 1.14× at 4k and nothing at 6k. The chart shows why: acceptance of the first draft token holds up to 4k, but the second and third positions, which build on the draft head's own guesses, collapse from about 0.75 to 0.16 and 0.07 between 2k and 4k. Our best explanation, low confidence, is that the draft head was trained on shorter contexts; we did not check its training length. The practical rule: gate speculative decoding on prompt length, off beyond roughly 4k tokens, or use a long-context draft head.

## Slide 10: Accuracy: The Two FP8 Changes Interact

(~1.5 min) Output quality. Our first, trimmed evaluation excluded each FP8 change on its own: FP8 weights for losing 3 points on HumanEval's 164 problems, FP8 KV cache for 2.8 points on 500 GSM8K questions. Its confidence intervals were wider than our 1-point gate, so in phase 2 we re-tested on the full benchmarks, pairing each answer with Default's answer to the same question, with a McNemar test. Result: neither FP8 change alone can be distinguished from BF16; the intervals are still too wide to confirm or rule out a one-point drop, so the original exclusions are not supported by the data. Together they lose 2.58 GSM8K points, significant at p = 0.03; that's the red bar. EAGLE-3 changed 47 of 1,319 answers, not significantly, and no HumanEval result. Our interpretation: each FP8 change adds a small rounding error that compounds over multi-step arithmetic. Caveat: the FP8-plus-spec combinations were not paired-tested, so the higher speedups you get by also allowing FP8 weights, which we call policy B, up to 3.3×, are only an upper bound.

## Slide 11: What Limits Throughput: GEMMs at a Power Cap

(~1 min) What is the binding constraint? The harness's first-order rule labelled every winner "per-step overhead", meaning CPU-side. We profiled instead. In all 18 traces the GPU was executing kernels 96 to 99% of the time, mostly cuBLAS matrix multiplies, and every phase-2 run hit the 400-watt power-cap throttle. The chart shows Default's clock falling with load: about 1,500 MHz at 2 in flight, under 900 at 128. Our reading: the limit is GEMM work at power-capped clocks, and an optimization helps in proportion to the matrix-multiply work it removes, which is exactly what FP8, speculation and prefix caching do.

## Slide 12: Can You Trust the Numbers?

(~1 min) Given the contamination story, how do we know the remaining numbers are right? Four checks; the first three run on any laptop. One: a check script recomputes every number in the report from the exported run data, 289 of 289 pass; it also caught an error in our own earlier claims about within-cell spread. Two: the harness's unit tests run without a GPU, 68 of 69 pass, the last needs the model tokenizer. Three: a dry run executes the full seven-optimization plan on a simulated engine, 1,536 runs, including injected failures, so the code for the two unmeasured optimizations is exercised. Four, on the GPU node: we re-ran one cell ten times with the other GPU busy and idle; no difference, so running two cells at once was safe. And the 40 uncontaminated phase-1 runs that were re-run match within 2.2%.

## Slide 13: What to Do with This

(~1 min) Five practical rules, all for our workload. Turn prefix caching on whenever prompts repeat; it costs nothing when they don't. Turn EAGLE-3 on for prompts up to about 2 to 4k tokens, especially at low load, and off beyond that. Use FP8 weights, which pair well with speculation for throughput, though that pairing is not yet paired-tested for accuracy; and don't combine them with FP8 KV cache when multi-step reasoning accuracy matters. Don't bother tuning max_num_seqs while your in-flight count stays below 256. And when estimating a combination, multiply the single speedups.

## Slide 14: Limitations and Next Steps

(~1 min) Limitations: one model, one draft head and synthetic prompts, so absolute numbers for speculation and prefix caching may not transfer to real traffic. 16 cells were lost to contamination and never re-run. The FP8-plus-speculation combinations weren't paired-tested for accuracy. And two of the seven optimizations remain unmeasured. Next steps in priority order: paired accuracy for those combinations, which decides whether the up-to-3.3× configurations are usable; re-run the 16 lost cells with exclusive GPUs and a per-run check for foreign GPU memory; a prompt-length gate for speculation; chunked prefill and parallelism on a node with at least four GPUs; and a real-traffic workload. Thank you, happy to take questions.