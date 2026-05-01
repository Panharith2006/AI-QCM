# Layer 2 - CLAHE Impact Analysis

## Your Custom Analysis

### Observations
_Add your observations about CLAHE performance:_
- How well does it handle shadows?
- Does it amplify noise in certain regions?
- What happens with over-exposed areas?

### Parameter Tuning

**Current Parameters:**
```
clip_limit = 2.0
tile_size = (8, 8)
```

**Experiments:**
- [ ] Tested clip_limit = 1.0 (less aggressive)
- [ ] Tested clip_limit = 3.0 (more aggressive)
- [ ] Tested tile_size = (6, 6) (smaller tiles, more local)
- [ ] Tested tile_size = (10, 10) (larger tiles, more global)

**Results:**
_Document which parameters work best for your test sheets_

### Edge Cases
- How does it handle heavily creased sheets?
- What about sheets with watermarks or background patterns?
- Does color variation affect the enhancement?

### Recommendation
_Your final recommendation for this layer_
