# ReAct implementation using Wikipedia and Guidance

This is an attempt to implement the [ReAct](https://arxiv.org/pdf/2210.03629.pdf) paper
using [guidance](https://github.com/guidance-ai/guidance) and the Wikipedia api.

## General notes and observations

### Julius Caesar's age when he was assassinated

ReAct prompting struggles when the model's next action is not a search, but a calculation,
comparison, reasoning, etc. By forcing it to do a search, the model often gets side tracked
from its correct original thought. For example, when I asked the model "What was Julius
Caesar's age when he was assassinated?", the model correctly figured out his birth and
death dates, and planned to then calculate his age. Instead, it used the lookup action
again to find Julius Caesar's age when he died, which is not explicitly written in the
Wikipedia page for Julius Caesar. Causing the model to get completely sidetracked. The
final answer was then sometimes correct, sometimes completely wrong.

Doing some tweaking to the prompt, and adding an "Infer"/"Nop" action, the model managed
to reach the perfect answer of 55 years, 8 months, and 3 days.

The "Infer" action told the model specifically to Infer information from previous thoughts
and actions. While the "Nop" action let the model perform no action, thereby allowing
it to infer information in the next thought. Both approaches were not stable, though.

## Implementation details

### Search

The paper specifies a very basic search strategy. I modified the lookup action to use
semantic search instead of basic string matching. There is also a bm25 lookup option
implemented there, but it is not used. To use bm25, replace the lookup action in
the `State#act` method to use `lookup_bm25`.

### Model

I used [ehartford/dolphin-2.1-mistral-7b](https://huggingface.co/ehartford/dolphin-2.1-mistral-7b). I also
used [HuggingFaceH4/zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha), this model
doesn't run on M1 Macs, because it uses BFloat16 which is not supported by MPS (AFAIK), so I only used it in Colab.

### Wikipedia

I used both [Wikipedia-API](https://pypi.org/project/Wikipedia-API/) and
[wikipedia](https://pypi.org/project/wikipedia/). The wikipedia library doesn't support setting
the user-agent, but the Wikipedia-API library doesn't support search. So I used wikipedia only
for the search query, and Wikipedia-API for fetching the page. Not really important and can probably
be cleaned to only use one.

I did not handle tables and images at all, they were not supported natively by the libraries (though
I didn't search very hard).
