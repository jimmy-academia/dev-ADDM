# GateInit Prompt Example (Illustrative)

This is a standalone example to show the **style** of gates GateInit should produce.
It is **not** tied to allergy or any specific policy. Do not copy wording into
actual prompts verbatim; use it as a shape reference.

## Example Topic: Billing / Charges

Primitive:
- primitive_id: BillingIssue_clause_1
- clause_quote: "- customer was charged twice or received an unexpected extra charge"

Example gate outputs (diverse aspects, review language):

```
{
  "primitives": [
    {
      "primitive_id": "BillingIssue_clause_1",
      "pos_bm25_gates": [
        ["charged twice", "double charged"],
        ["overcharged", "incorrect bill"],
        ["extra fee", "unexpected charge", "mysterious charge"]
      ],
      "neg_bm25_gates": [
        ["pricey but", "worth it"],
        ["good value", "reasonable price"],
        ["expensive", "cheap"]
      ],
      "pos_emb_gates": [
        "They charged my card twice and I had to dispute it.",
        "There was an extra fee on the receipt that nobody mentioned.",
        "I was overcharged compared to the listed price and they wouldn't fix it."
      ],
      "neg_emb_gates": [
        "A bit pricey, but the quality is great.",
        "Not cheap, but you get what you pay for.",
        "Great value for the money."
      ]
    }
  ]
}
```
