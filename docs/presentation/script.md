# Presentation Script: Combining LLM Serving Optimizations (Final Presentation)

Project 24, Group 2 · Panshul Jindal, Mohd Shozab · 30 slides, about 21 minutes at 150 words per minute, plus questions.
The same text is in each slide's speaker notes in `LLM_Serving_Final_Presentation.pptx`.

## Slide 1: Combining Infrastructure-Level Optimizations for LLM Serving

Hi everyone. We're presenting project 24, on combining infrastructure-level optimizations for LLM serving. The question we started with is a practical one: if you are running vLLM and you can turn on FP8, speculative decoding, prefix caching and a few other things, which of them actually give you the speedup, under what kind of traffic, and where do they stop helping? The circles are the seven optimizations; the two dashed ones we implemented but didn't measure. We'll go through the background, how we set up the study, what we measured, one thing that went badly wrong in the middle and how we fixed it, and then the results.

## Slide 2: Background: two phases, two bottlenecks

A quick reminder of why serving is a systems problem. Prefill runs over the whole prompt at once, so it has plenty of arithmetic and is compute-bound. Decode produces one token per step, and every step re-reads the weights and the KV cache for very little work, so it is limited by memory bandwidth. The memory a batch needs is roughly the weights plus one KV cache per request, so how much KV cache fits decides how many requests you can batch, and that decides throughput. A lot of the well-known serving techniques are attacks on exactly this.

## Slide 3: Motivation: published speedups don't add up

This is the motivation from our progress report. Every optimization paper measures against its own baseline: vLLM against Orca, Sarathi-Serve against vLLM, SGLang against an older vLLM. So you can't multiply their numbers together. On top of that, the gains depend on the workload and on each other. Speculative decoding can make latency worse at high load, and on a quantized 70B model it added only 1.8 times. Prefix caching only helps when prompts actually repeat. If you're deploying a model, you can't tell from the literature which of these to turn on together, and we didn't find a study that measures all seven against a single baseline.

## Slide 4: Related work: each technique on its own

Here is the related work, one row per optimization. Each one is well established on its own: vLLM's paging gave two to four times over Orca, chunked prefill raised capacity, SGLang's prefix reuse reports up to 6.4 times when the cache hits. The column on the right is the important one: each gain only holds in some conditions. And the numbers in the middle column are against different baselines, so they are not comparable with each other. The few controlled combination studies vary at most three options. Nobody varies all seven across the three workload dimensions, and that is the gap we went after.

## Slide 5: Problem statement and research question

The problem statement, in one sentence: improve latency and throughput by combining infrastructure-level optimizations, and find out which ones contribute, in which conditions, and where they stop working. We look at three regimes: prompt length, how many requests are in flight, and how much of the traffic shares a prefix. We move one at a time away from a centre point, which gives seven workload points. On the right are the concrete questions we used to organise the evaluation; the last one turned out to matter more than we expected.

## Slide 6: The seven optimizations and their on/off states

These are the seven optimizations, with their off state, their on state and the vLLM flag we use. The first three change a single engine instance. The last four depend on the workload. In Default every one of them is off, including prefix caching and chunked prefill, which vLLM normally switches on by itself; otherwise every gain we report would be understated. The last column is the honest one: we measured the first five. Chunked prefill and parallelism are implemented in the harness but we did not run them on GPUs, and I'll explain why shortly.

## Slide 7: Methodology (1/4): factorial design on one engine

How the study is designed. Everything runs on one engine and one model, and every optimization is a switch. At the centre point we run all 32 combinations of the five switches, so the attribution there is exact. At the six other points we run an even-parity half-fraction, 16 combinations each, which still lets us separate every main effect and every pairwise interaction. That's 128 cells, each repeated three times. With all seven optimizations the same design would have been 512 cells, and a full factorial at every point would have been 896.

## Slide 8: Methodology (2/4): value and attribution

How we score a configuration. Value is its throughput divided by Default's at the same point, but only if three gates pass: p99 time per output token under 100 milliseconds, p99 time to first token under 5 seconds, and no more than one point of accuracy lost on any task. Otherwise we count it as no benefit. To attribute the gain we fit main effects on log throughput, and for each pair we compute this interaction ratio. If it's one, the two speedups just multiply. Above one, they help each other; below one, they overlap. Shapley shares then split each winning configuration's gain among its parts, and the binding constraint comes from telemetry.

## Slide 9: Methodology (3/4): measurement harness

This is the harness we built around vLLM, Figure 3 of the report. A configuration manager turns a set of switches into exact vLLM flags, a server manager launches one engine per GPU, a workload generator makes a fixed trace, and a closed-loop load driver replays it while the telemetry collector samples the GPU and the engine. Quality checks run against the same server. Everything lands in an append-only results store. The analysis on the right runs offline, which is why the numbers in this talk can be re-checked without a GPU.

## Slide 10: Methodology (4/4): how one run works

A bit more on how a single run works. The prompts are synthetic random words, but sent as token IDs so the engine sees exactly the length we asked for, and reuse comes from four shared prefixes. Every configuration at a point replays the same saved trace. The load driver keeps exactly C requests in flight, throws away a 30-second warm-up and measures over the next two minutes. On the reliability side, a configuration that fails to start is written down as a result, and jobs pick up where they left off, which mattered because the cluster kills jobs after eight hours. We also test the whole pipeline without GPUs on a simulated engine.

## Slide 11: From plan to what we actually ran

This is where the plan and reality differ. We planned seven optimizations and 512 cells; we ran five and 128. We planned around an Ampere GPU, which has no FP8 arithmetic, so FP8 would have only saved memory; in the end we got FP8-capable GPUs, first a four-GPU RTX 6000 Ada node, which was replaced on 5 November by a Slurm node with two H100s. We swapped DCGM and Nsight for tools that run without admin rights. And we added a second phase to investigate things that looked odd in the first one. That second phase turned out to be important.

## Slide 12: Why we measured 5 of the 7 optimizations

Why only five. Parallelism is the big one. It needs more than one GPU per cell, and in our harness every layout, including Default's tensor-parallel layout, has to use all the GPUs of a cell. So with parallelism in the design every cell would occupy the whole node, and we'd lose the ability to measure two cells at once. Add the fourfold increase in cells, and the factorial alone goes from 11.5 hours to roughly 90, about our whole budget. Chunked prefill mostly helps tail latency rather than throughput, and it needs its own sweep. So we dropped both, kept them in the code, and spent the time on the follow-up phase instead.

## Slide 13: Experimental setup

The setup. Two H100 NVL GPUs on a shared Slurm node, capped at 400 watts. vLLM 0.31 with Llama 3.1 8B Instruct and an EAGLE-3 draft head. Phase one was 384 runs over about eleven and a half hours on 7 November. Phase two, four days later, re-ran 60 cells with detailed telemetry, took 18 profiler traces and re-tested accuracy on the full benchmarks. Repetitions were tight, with a coefficient of variation of at most about one and a half percent on the cells that matter. Keep in mind that the GPUs were shared with other users; that comes back in a few slides.

## Slide 14: Results (1/12): Default versus best configuration

The headline result. For each workload, the best configuration that passes every gate reaches between 1.39 and 2.62 times Default's throughput. Speculative decoding plus prefix caching wins at six of the seven points; at 90% reuse the winner also has tuned batching on, but batching itself adds nothing. The exception is long, 8k-token prompts, where EAGLE-3 stops helping and prefix caching with batching wins instead. The smallest gain is with no prompt reuse at all, and the largest is when most requests share a prefix.

## Slide 15: Results (2/12): latency gates

Before we trust a speedup, the latency gates have to hold. The grey bars are Default, the blue bars are the winner, and the dashed lines are the gates. Nothing comes close: no cell anywhere failed a latency gate, so every exclusion in this study was for accuracy. Speculation brings time to first token down a lot, and it can add a little to time per output token, for example at 0% reuse from about 18 to about 23 milliseconds, which is still nowhere near the 100-millisecond gate.

## Slide 16: Results (3/12): a hidden process on GPU 0

This is the thing that went wrong. When we re-ran some phase-one cells in phase two, they didn't match. Each dot here is one phase-one run divided by its clean re-run. Until 11:24 on 7 November everything sits at one. After that, every run on GPU zero drops to between 0.56 and 0.69 of its real speed, while GPU one stays exactly at one. NVML had recorded the signature: four gigabytes of memory that weren't ours, a higher clock and lower power. Another user's process had started sharing the GPU. 71 runs, 18 percent of phase one, were affected, and the worst part is that it was easy to miss: the repetitions varied only 0.2 to 3.8 percent, about ten times the spread of clean cells, but we weren't checking for that, so nothing warned us.

## Slide 17: Results (4/12): correcting the contaminated runs

How we fixed it. We didn't divide by a correction factor, because the slowdown wasn't constant: it ranged from 0.56 to 0.69. Instead we replaced the eight cells phase two had re-run, excluded sixteen we couldn't re-run, and added sixteen cells that phase two measured at the standard points. The effect on the headline is large in two places: what first looked like 4.5 times at 90% reuse is really 2.62, and 2.64 at high concurrency is really 1.67. The uncontaminated runs that we did re-run agree within 2.2 percent, which is our evidence that the rest is sound.

## Slide 18: Results (5/12): which optimization matters where

This is each optimization's average multiplier at each workload, fitted on the clean data. The model explains more than 99.7 percent of the variance everywhere. FP8 weights are the steadiest: between 1.22 and 1.34 at every point. EAGLE-3 is the biggest lever when the GPU has spare capacity, 2.31 times with two requests in flight, but on 8k-token prompts it gives nothing. Prefix caching follows reuse; with no reuse it's flat, and it doesn't cost anything either. FP8 KV cache only helps where the KV cache is large. Tuned batching never helps, because the engine's default limit of 256 is already above every in-flight count we tested.

## Slide 19: Results (6/12): all 32 configurations at the centre

Here is every one of the 32 configurations at the centre, ranked. Nothing is slower than Default. The four fastest all combine FP8 weights with speculation and prefix caching, around 2.4 to 2.5 times. But they're grey, which means they failed our original accuracy gate, so they don't count. The fastest blue bar, spec plus prefix at 1.82, is the official winner. Two of them pair FP8 weights with FP8 KV, which the accuracy re-test shows does lose accuracy; the other two pair FP8 weights with speculation, which we never re-tested, so they stay unproven. I'll show the accuracy results later.

## Slide 20: Results (7/12): who earns the gain

This splits each winner's gain between its parts using Shapley values. At the centre it's roughly sixty-forty between speculation and prefix caching, and that one is exact because we measured every subset there. The pattern across workloads makes sense: when there is little reuse or the GPU is under-used, speculation does almost all the work; when many requests share a prefix or the batch is big, prefix caching does. The values marked approximate come from the fitted model, because the half-fraction designs don't measure every subset.

## Slide 21: Results (8/12): combinations multiply

One of our main questions was whether optimizations add, multiply or cancel. Each dot is a combination; on the x axis is the product of its single-optimization speedups, on the y axis what we measured. If they simply multiplied, the dots would sit on the dashed line. None of the fifty falls meaningfully below it, so in every combination we could test, at four of the seven workloads, they never cancelled. Eighteen are clearly above it, and every one of those contains speculative decoding; the biggest is 31 percent above. For FP8 weights with speculation, our interpretation is that each engine step does a much bigger matrix multiply, 928 tokens instead of 284, which is where FP8 should help most.

## Slide 22: Results (9/12): prefix caching and speculation

The strongest interaction we found is between prefix caching and speculation. Alone, prefix caching gives 1.52 times at 90% reuse. On top of EAGLE-3 it gives 1.85. The telemetry points to where it comes from: prefix caching shortens each request's decode phase by 27 percent without speculation and by 45 percent with it. Our explanation, and it is an interpretation, is that vLLM mixes prompt work and decoding in the same steps; speculation makes the decoding part smaller, so the prompt work that caching removes becomes a bigger share. The effect is flat up to 25% reuse and clearly there at 50%; we didn't pin down where in between it starts.

## Slide 23: Results (10/12): where EAGLE-3 stops helping

This is where an optimization stops working. EAGLE-3 gives 1.35 times at 2k-token prompts, 1.14 at 4k and nothing at 6k. The left plot shows why: the first draft token is still accepted most of the time up to 4k, but the second and third positions, which build on the draft head's own guesses, collapse between 2k and 4k. Our best guess is that the draft head was trained on shorter contexts, but we didn't check its training length, so treat that as a hypothesis. The practical rule is simple: switch speculation off for prompts beyond roughly 4k tokens, or get a draft head trained for long contexts.

## Slide 24: Results (11/12): output quality

On output quality. Our first evaluation used trimmed benchmarks, 500 GSM8K questions, and on those FP8 weights and FP8 KV each seemed to lose more than a point, so we excluded them. But with that few questions the confidence interval is wider than the gate. So in phase two we re-ran the full benchmarks and compared each answer with Default's answer to the same question. For each FP8 change on its own the result is inconclusive: no significant drop, but the interval is too wide to rule out losing a point. Together, they lose 2.58 points on GSM8K, and that one is statistically significant. Speculative decoding changed 47 answers out of 1,319, which isn't significant. We didn't re-test the FP8-plus-speculation combinations, so those faster configurations remain unproven.

## Slide 25: Results (12/12): what limits throughput

What actually limits throughput? The harness has a simple rule that labelled every winner as limited by per-step overhead, meaning CPU-side work. We didn't trust that, so we profiled. In all 18 traces the GPU was running kernels between 96 and 99 percent of the time, mostly matrix multiplies, and every run in phase two hit the 400-watt power cap. You can see the clock dropping as load goes up, from about 1,500 megahertz with two requests to under 900 with 128. Our reading is that the limit is matrix-multiply work at a power-capped clock, so an optimization helps roughly in proportion to how much of that work it removes.

## Slide 26: Can the numbers be trusted?

Given what happened with the shared GPU, we wanted solid evidence for everything else. This plot is one cell that looked noisy in phase one. In phase two we ran it five times with the other GPU busy and five times with it idle, and the results are the same, within one percent. The noise in phase one was the interference. Beyond that, a check script recomputes every number in the report from the exported data, and all 289 match; it actually caught two mistakes in our earlier drafts. The harness's unit tests run without a GPU, and a dry run exercises the full seven-optimization plan, including the two we didn't measure.

## Slide 27: Key findings

To sum up what we found. The best configurations give between 1.39 and 2.62 times Default's throughput without breaking any latency gate, and speculation plus prefix caching is the winner almost everywhere. In every combination we could test, optimizations didn't cancel each other; their speedups multiply, and with speculation they do a bit better than multiplying. EAGLE-3 stops being useful beyond about 4k-token prompts. The two FP8 changes together cost accuracy; on their own, the test couldn't tell either way. Our reading is that throughput is limited by matrix-multiply work at the power cap. And the one we'd most like other people to remember: a process we didn't own silently slowed 18 percent of our runs, and only re-running cells in a new session revealed it.

## Slide 28: What we would tell someone deploying vLLM

If someone asked us how to configure vLLM based on this, these are the five things we'd say, with the caveat that it's one model and synthetic prompts. Prefix caching: turn it on whenever prompts repeat, there's no downside we could measure. EAGLE-3: use it for short and medium prompts, especially when load is low, and turn it off past about 4k tokens. FP8 weights work well and pair nicely with speculation for speed, though we haven't tested that pair's accuracy yet, and be careful adding FP8 KV cache on top if your task needs multi-step reasoning. Don't spend time tuning the batch limit unless you have more than 256 requests in flight. And to estimate a combination, multiply.

## Slide 29: Limitations and future work

The limitations, honestly. It's one model with one draft head and synthetic prompts, so the absolute numbers for speculation and prefix caching might not carry over to real traffic. Two of the seven optimizations were never measured on GPUs. The configurations that combine FP8 with speculation look faster, up to 3.3 times, but we haven't re-tested their accuracy. And the shared node doesn't isolate GPUs between users. The future work follows from that: re-test those configurations, run on exclusive GPUs with a check for other users' processes, add a prompt-length gate for speculation, measure the two missing optimizations on a bigger node, and try real traffic.

## Slide 30: Thank you! Questions?

That's our project. Thank you for listening; we're happy to take questions, and the report, the harness and the check script are all in the repository if you want to look at the details.
