# G2: Social Context Fit

*Perspective A: Customer*
*Goal: Synthesize ambiance for specific social goals.*

## Topics

1. **Romance** - Date night and romantic dining
2. **Business** - Professional/business dining
3. **Group** - Family and group dining

---

## L0 Primitives

### Romance

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `DATE_CONTEXT` | bool | true/false | Review mentions date/romantic context |
| `AMBIANCE_RATING` | enum | poor/fair/good/excellent | Overall romantic ambiance |
| `NOISE_LEVEL` | enum | loud/moderate/quiet/intimate | Noise level assessment |
| `PRIVACY_LEVEL` | enum | none/low/moderate/high | Privacy for conversation |
| `SERVICE_STYLE` | enum | rushed/normal/attentive/romantic | Service appropriateness |
| `OUTCOME` | enum | negative/neutral/positive/memorable | How the date went |

### Business

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `BUSINESS_CONTEXT` | bool | true/false | Review mentions business context |
| `PROFESSIONALISM` | enum | poor/adequate/good/excellent | Professional atmosphere |
| `NOISE_LEVEL` | enum | loud/moderate/quiet | Suitable for discussion |
| `SERVICE_SPEED` | enum | slow/normal/efficient | Efficiency for busy professionals |
| `PRIVACY_SUITABLE` | enum | poor/fair/good | Privacy for business talk |
| `OUTCOME` | enum | negative/neutral/positive | Meeting success |

### Group

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `GROUP_CONTEXT` | bool | true/false | Review mentions group dining |
| `GROUP_SIZE` | enum | small_4_6/medium_7_10/large_10_plus | Size of group |
| `ACCOMMODATION` | enum | poor/adequate/good/excellent | How well group was handled |
| `SPLIT_CHECK` | enum | refused/difficult/easy | Bill splitting experience |
| `WAIT_TIME` | enum | excessive/long/reasonable/quick | Wait for large party |
| `OVERALL_SUCCESS` | enum | disaster/mixed/success/great | Overall group experience |

---

## Keywords for Restaurant Selection

### Romance
```
date night, romantic, anniversary, intimate, proposal, special occasion,
candlelit, cozy, romantic dinner, date spot, couples, valentines,
romantic atmosphere, perfect for dates
```

### Business
```
business lunch, business dinner, meeting, client, professional,
interview, work lunch, corporate, business casual, quiet enough,
expense account, power lunch, networking
```

### Group
```
large group, party, birthday, celebration, group dinner, big table,
reservation for, family dinner, group of, private room, family friendly,
kids, children, high chair, accommodate group
```
