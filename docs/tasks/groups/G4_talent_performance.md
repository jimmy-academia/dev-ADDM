# G4: Talent & Performance

*Perspective B: Business Owner*
*Goal: Evaluate human capital and environment.*

## Topics

1. **Server** - Front-of-house staff quality
2. **Kitchen** - Food execution and chef skill
3. **Environment** - Ambiance and physical comfort

---

## L0 Primitives

### Server

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `SERVER_MENTIONED` | bool | true/false | Review mentions server/waiter |
| `ATTENTIVENESS` | enum | absent/poor/adequate/good/excellent | How attentive was service |
| `FRIENDLINESS` | enum | rude/cold/neutral/friendly/warm | Demeanor of staff |
| `KNOWLEDGE` | enum | none/poor/adequate/good/excellent | Menu/wine knowledge |
| `ERROR_MADE` | bool | true/false | Did server make a mistake |
| `NAMED_POSITIVELY` | bool | true/false | Mentioned server by name positively |

### Kitchen

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `FOOD_ISSUE_MENTIONED` | bool | true/false | Review mentions food issue |
| `ISSUE_TYPE` | enum | temperature/doneness/taste/presentation/other | Type of food issue |
| `SEVERITY` | enum | none/minor/major/inedible | How bad was the issue |
| `CONSISTENCY` | enum | inconsistent/variable/consistent | Consistency of food quality |
| `CHEF_SKILL` | enum | poor/average/good/excellent | Perceived chef skill |
| `SENT_BACK` | bool | true/false | Was food sent back |

### Environment

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `ENVIRONMENT_MENTIONED` | bool | true/false | Review mentions environment |
| `CLEANLINESS` | enum | dirty/fair/clean/spotless | Cleanliness level |
| `NOISE_LEVEL` | enum | deafening/loud/moderate/quiet | Noise assessment |
| `COMFORT` | enum | uncomfortable/fair/comfortable/very_comfortable | Seating comfort |
| `TEMPERATURE` | enum | too_cold/cold/comfortable/hot/too_hot | Room temperature |
| `AMBIANCE_FIT` | enum | poor/fair/good/excellent | Does ambiance fit concept |

---

## Keywords for Restaurant Selection

### Server
```
waiter, waitress, server, our server, service was, attentive, ignored,
rude server, friendly staff, helpful, knowledgeable, forgot, remembered,
amazing service, terrible service, slow service, efficient
```

### Kitchen
```
overcooked, undercooked, burnt, raw, cold food, perfect temperature,
chef, cook, kitchen, food quality, presentation, plating, consistency,
sent back, inedible, delicious, bland, seasoning
```

### Environment
```
loud, noisy, quiet, atmosphere, ambiance, decor, comfortable, cramped,
seating, table, booth, patio, temperature, AC, air conditioning,
too cold, too hot, music, lighting, clean, dirty
```
