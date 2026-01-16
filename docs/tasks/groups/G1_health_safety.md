# G1: Health & Safety

*Perspective A: Customer*
*Goal: Deduce safety from implicit signals.*

## Topics

1. **Allergy** - Allergen safety (peanut, shellfish, tree nut)
2. **Dietary** - Dietary restriction accommodation (gluten, dairy, vegan)
3. **Hygiene** - Cleanliness and food safety

---

## L0 Primitives

### Allergy

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `ALLERGY_MENTIONED` | bool | true/false | Review mentions allergy context |
| `INCIDENT_SEVERITY` | enum | none/mild/moderate/severe | Severity of any allergic reaction |
| `ALLERGEN_TYPE` | enum | peanut/tree_nut/shellfish/other | Type of allergen involved |
| `ACCOUNT_TYPE` | enum | firsthand/secondhand/hearsay | How direct is the account |
| `STAFF_RESPONSE` | enum | none/dismissive/helpful/proactive | How staff handled allergy request |
| `ASSURANCE_CLAIM` | bool | true/false | Did restaurant claim to accommodate |

### Dietary

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `DIETARY_MENTIONED` | bool | true/false | Review mentions dietary restriction |
| `RESTRICTION_TYPE` | enum | gluten/dairy/vegan/vegetarian/other | Type of dietary restriction |
| `ACCOMMODATION_LEVEL` | enum | none/poor/adequate/excellent | How well they accommodated |
| `MENU_LABELING` | enum | none/vague/clear/detailed | Quality of menu labeling |
| `INCIDENT_OCCURRED` | bool | true/false | Did a dietary incident occur |
| `STAFF_KNOWLEDGE` | enum | none/poor/adequate/excellent | Staff knowledge of ingredients |

### Hygiene

| Primitive | Type | Values | Description |
|-----------|------|--------|-------------|
| `HYGIENE_MENTIONED` | bool | true/false | Review mentions hygiene |
| `ISSUE_SEVERITY` | enum | none/minor/major/severe | Severity of hygiene issue |
| `ISSUE_TYPE` | enum | cleanliness/pest/food_safety/bathroom/other | Type of issue |
| `ACCOUNT_TYPE` | enum | witnessed/inferred/reported | How issue was observed |
| `STAFF_RESPONSE` | enum | none/dismissive/addressed/proactive | Staff response to issue |
| `ILLNESS_REPORTED` | bool | true/false | Did reviewer report getting sick |

---

## Keywords for Restaurant Selection

### Allergy
```
peanut, nut allergy, tree nut, allergic reaction, anaphylaxis, epipen,
shellfish allergy, allergy friendly, allergy menu, cross contamination,
nut-free, peanut-free, allergy safe
```

### Dietary
```
gluten free, gluten-free, celiac, dairy free, dairy-free, lactose,
vegan, vegetarian, plant based, dietary restrictions, special diet,
gluten friendly, dairy alternative, vegan options
```

### Hygiene
```
dirty, filthy, roach, cockroach, bug, hair in food, food poisoning,
got sick, unsanitary, health code, cleanliness, gross, disgusting,
bathroom dirty, kitchen dirty, health department
```
