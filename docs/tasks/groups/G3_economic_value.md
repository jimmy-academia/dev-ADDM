# G3: Economic Value Assessment

*Perspective A: Customer*
*Goal: Calculate "True Cost" beyond the menu price.*

## Topics

1. **Price-Worth** - Value perception including deals
2. **Hidden Costs** - Unexpected charges
3. **Time-Value** - Is the wait worth it

---

## L0 Primitives

### Price-Worth

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `PRICE_MENTIONED` | bool | true/false | Review discusses price |
| `PRICE_PERCEPTION` | enum | ripoff/overpriced/fair/good_value/steal | Value perception |
| `PORTION_SIZE` | enum | tiny/small/adequate/generous | Portion assessment |
| `QUALITY_FOR_PRICE` | enum | poor/fair/good/excellent | Quality relative to cost |
| `DEAL_MENTIONED` | bool | true/false | Mentions happy hour, special, etc. |
| `WOULD_RETURN_PRICE` | enum | never/maybe/yes/definitely | Return intent based on price |

### Hidden Costs

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `HIDDEN_COST_MENTIONED` | bool | true/false | Mentions unexpected charges |
| `COST_TYPE` | enum | service_charge/auto_gratuity/corkage/split_fee/other | Type of hidden cost |
| `DISCLOSURE` | enum | none/fine_print/verbal/clear | How well it was disclosed |
| `AMOUNT_IMPACT` | enum | minor/moderate/significant | Impact on total bill |
| `REACTION` | enum | upset/annoyed/neutral/understanding | Customer reaction |
| `BILL_SURPRISE` | bool | true/false | Was bill higher than expected |

### Time-Value

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `WAIT_MENTIONED` | bool | true/false | Review mentions waiting |
| `WAIT_TYPE` | enum | seating/food/check/other | What they waited for |
| `WAIT_DURATION` | enum | none/short/moderate/long/excessive | How long the wait |
| `WAIT_JUSTIFIED` | enum | not_worth_it/barely/worth_it/definitely_worth_it | Was wait worth it |
| `RESERVATION_HONORED` | enum | no/delayed/yes | Did reservation work |
| `WOULD_WAIT_AGAIN` | enum | never/maybe/yes/definitely | Would wait again |

---

## Keywords for Restaurant Selection

### Price-Worth
```
expensive, overpriced, pricey, worth it, good value, cheap, affordable,
portion size, small portions, generous portions, bang for buck,
happy hour, lunch special, deal, discount, prix fixe, tasting menu
```

### Hidden Costs
```
service charge, auto gratuity, automatic tip, corkage fee, split fee,
hidden fee, surprised by bill, didn't expect, extra charge, added fee,
fine print, not disclosed, tip included, mandatory tip
```

### Time-Value
```
wait time, long wait, waited forever, worth the wait, not worth waiting,
hour wait, reservation, no reservation, walk in, wait list, seated late,
took forever, slow service, quick service, fast
```
