# ReAct implementation using Wikipedia and Guidance

This is an attempt to implement the [ReAct](https://arxiv.org/pdf/2210.03629.pdf) paper
using [guidance](https://github.com/guidance-ai/guidance) and the Wikipedia api.

## Implementation details

### State

The model's action affect a state object, that keeps track of the currently opened
Wikipedia article.

### Chunking

The article text (ignoring tables and images) is chunked using the embedder's
tokenizer. This might be an issue if you also use rerank, since they use different
tokenizers. Since the chunks size I used was at most 256 tokens, and the max_len of
both models is 512, this wasn't an issue.

### Search

The paper specifies a very basic search strategy. I modified the lookup action to use
semantic search instead of basic string matching. There is also a bm25 lookup option
implemented there, but it is not used. To use bm25, replace the lookup action in
the `State` init call to use `bm25`. To use a reranker after the semantic search,
set `rerank=True`.

### Infer action

Another action was added, other than `Search`, `Lookup`, and `Finish`, I added an `Infer` action.
This proved valuable for tasks requiring calculations, and where the answer to the
question was not explicitly in the text. More on the `Infer` action [below](#infer-how-old-was-julius-caesar-when-he-died).

### Model

I used [ehartford/dolphin-2.1-mistral-7b](https://huggingface.co/ehartford/dolphin-2.1-mistral-7b).
I also used [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha),
this model doesn't run on M1 Macs, because it uses `BFloat16` which is not supported by
MPS (AFAIK), so I only used it in Colab.

### Wikipedia

I used both [Wikipedia-API](https://pypi.org/project/Wikipedia-API/) and
[wikipedia](https://pypi.org/project/wikipedia/). The wikipedia library doesn't support setting
the user-agent, but the Wikipedia-API library doesn't support search. So I used wikipedia only
for the search query, and Wikipedia-API for fetching the page. Not really important and can probably
be cleaned to only use one. I'd prefer to specify the user-agent for all requests.

I did not handle tables and images at all, they were not supported natively by the libraries (though
I didn't search very hard).

## General notes and observations

### Infer: How old was Julius Caesar when he died?

ReAct prompting struggles when the model's next action is not a search, but a calculation,
comparison, reasoning, etc. By forcing it to do a search, the model often gets side tracked
from its correct original thought. For example, when asked "How old was Julius
Caesar when he died?", the model correctly figured out his birth and death dates, and
planned to then calculate his age. Instead, it used the lookup action again to find Julius
Caesar's age when he died, which is not explicitly written in the Wikipedia page for
Julius Caesar. Causing the model to get completely sidetracked. The final answer was then
sometimes correct, sometimes completely wrong.

Doing some tweaking to the prompt, and adding an "Infer" action, the model managed
to reach the perfect answer of 55 years, 8 months, and 3 days. Sadly I forgot the
prompt and wasn't able to replicate this result.

However, if the calculations performed during inference get too long, the model again
gets sidetracked, hallucinating facts like "to convert from BC to AD we need to add
100 years". The current Infer instruction still gets a close answer "56 years", but
doesn't easily get sidetracked due to long calculations.

Using a more powerful model will probably solve this issue, and encouraging the model
to calculate more verbosely would be stable.

### Prompt -> output determinism

Changing the prompt can cause unexpected changes in the output of the model, making it
seem "non-deterministic", even thought it technically isn't with `do_sample=False`.

But, slight tweaks to the prompt seem to nudge the model between different "branches".
The output does change, but toggles between a few (sometimes even just 2) sentences
or even paragraphs. Not sure what this means, but I found it interesting.

Using neural network interpretability tools like [Captum](https://github.com/pytorch/captum),
it may be possible to see which tokens caused the model to choose between the "branches".
Maybe by checking the output of the first attention layer for the first token that
starts the diverging branches?

### Chat

Using ReAct in a chat model can work, but the previous responses can affect the individual
result to each new question. Also, after a few responses the previous answers can leak
into the thoughts of the current question. When this did happen the final answer was
still correct, it just also contained irrelevant information from previous questions.
