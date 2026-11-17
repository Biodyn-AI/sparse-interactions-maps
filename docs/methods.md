# What we measure

For a recipient input and a source input, copy selected internal outputs from the source run into the recipient run. For two parts `i` and `j`, compare four runs with the same fixed background changes:

```text
interaction = output(change i and j) - output(change i)
            - output(change j) + output(change neither)
pair strength = mean squared interaction over inputs and output coordinates
```

A zero interaction means those changes add in the measured output. A nonzero interaction does not isolate a direct connection, identify a direction, or prove that information travels between these two parts. Changing either part also affects later computation.

## Four distinct checks

1. **Repeatability:** select pairs on discovery inputs; evaluate score coverage and ranking on fresh confirmation inputs.
2. **Pair-specific structure:** compare the table with a product `a[i] * b[j]`. Each table has its own fitted values; earlier/later roles can differ. A good fit summarizes a measured table, not a validated causal importance for every part or a cheap acquisition algorithm.
3. **Reference connections:** compare direct links, pairs connected by longer paths, and pairs with no path in the known task circuit. Report label prevalence as well as ranking precision. Candidate pairs follow model order, but the interaction statistic itself is symmetric.
4. **Joint circuit behavior:** change a selected set together and, separately, keep it unchanged while replacing its complement. Check output projections, squared errors, and answer loss. A successful ranking does not replace these tests.

The manuscript gives exact formulas and thresholds. E31 uses name-answer margins in GPT-2; E32/E33 use class-centered answer logits in selected small benchmark models. This is a staged series of diagnostics with different interventions and cohorts, not one matched ablation sequence in every model.

## Scope and limitations

- Model cases and some parent blocks were selected with earlier knowledge. New confirmation inputs do not make the model sample random or blind.
- In E33, the fitted product uses the confirmation table itself; it is descriptive, not unseen-score prediction.
- Reference circuits are known task implementations, not necessarily the unique explanation of all full-model computation.
- Replacement states can create unusual combinations. Changing the replacement/normalization changes the test.
- Squaring loses effect signs. Strong score concentration does not imply corresponding task-performance retention.
- The algebraic examples illustrate standard facts; they are not claimed as new theorems.
- Passing a part-group test and failing a later connection-set test do not establish that parts inherently outperform connections: those tests use different settings and budgets.

See the [claim map](claim_map.md), original plans, and the manuscript's limitations before using a score map as a wiring diagram.
