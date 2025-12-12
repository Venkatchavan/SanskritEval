# Manual Annotation Guide for Sandhi Gold Test Set

## Overview

The gold test set (`sandhi_gold_test.jsonl`) contains automatically generated segmentations that need **manual verification and correction**. This guide explains how to annotate the data properly.

## What is Sandhi?

Sandhi (संधि) is the phonological process in Sanskrit where sounds at word boundaries merge according to specific rules. For example:
- रामः + अगच्छत् → रामोऽगच्छत् (visarga + vowel → o)
- तत् + च → तच्च (t + c → c doubled)

Our task is to identify where these boundaries occur in the fused form.

## Annotation Task

### Input Format (JSONL)

Each line contains one example:

```json
{
  "verse_id": "1.1",
  "fused": "धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।",
  "segmented": "धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय। ।",
  "confidence": 0.0,
  "is_gold": true
}
```

### Your Task

1. **Review `fused` field**: This is the original verse with sandhi
2. **Correct `segmented` field**: Add/remove spaces to mark word boundaries
3. **Update `confidence` to 1.0**: After you verify it's correct
4. **Keep `is_gold` as true**

### Annotation Principles

#### 1. Word Boundaries
Mark boundaries with spaces. For example:
```
Fused:      पाण्डवाश्चैव
Segmented:  पाण्डवाः च एव
            ↑       ↑  ↑
            word boundaries
```

#### 2. Compound Words (Samasa)
Sanskrit compound words should generally be kept together unless they're clearly separable:
```
धर्मक्षेत्रे  →  धर्मक्षेत्रे  (keep compound)
पाण्डवानाम्  →  पाण्डवानाम्  (keep inflection)
```

#### 3. Sandhi Examples

**Visarga Sandhi:**
```
Fused:      मामकाः पाण्डवाश्चैव
Segmented:  मामकाः पाण्डवाः च एव
            (मामकाः remains, पाण्डवाश् → पाण्डवाः + च)
```

**Vowel Sandhi:**
```
Fused:      रामोऽगच्छत्
Segmented:  रामः अगच्छत्
            (o + ' → aḥ + a)
```

**Consonant Sandhi:**
```
Fused:      तच्च
Segmented:  तत् च
            (tac → tat + ca)
```

#### 4. Punctuation
- Keep दण्ड (।) as sentence markers
- Double दण्ड (॥) marks verse end
- Add space after punctuation if needed

### Common Patterns in Bhagavad Gita

1. **Speaker attributions:**
   ```
   धृतराष्ट्र उवाच  →  धृतराष्ट्र उवाच (keep as is)
   सञ्जय उवाच      →  सञ्जय उवाच
   ```

2. **च (and) after words:**
   ```
   पाण्डवाश्चैव  →  पाण्डवाः च एव
   ```

3. **Common word endings with sandhi:**
   ```
   -श्च    →  -ः च
   -स्त्व   →  -स् त्व
   -ोऽ    →  -ः अ
   ```

## Annotation Workflow

### Step 1: Open the file
```bash
code data/benchmarks/sandhi_gold_test.jsonl
```

### Step 2: For each line

1. Read the `fused` field carefully
2. Check if the `segmented` field is correct
3. If incorrect, edit the `segmented` field to add proper word boundaries
4. Update `confidence` from `0.0` to `1.0`
5. Save the file

### Step 3: Example Correction

**Before:**
```json
{
  "verse_id": "1.1",
  "fused": "मामकाः पाण्डवाश्चैव",
  "segmented": "मामकाः पाण्डवाश्चैव",
  "confidence": 0.0,
  "is_gold": true
}
```

**After:**
```json
{
  "verse_id": "1.1",
  "fused": "मामकाः पाण्डवाश्चैव",
  "segmented": "मामकाः पाण्डवाः च एव",
  "confidence": 1.0,
  "is_gold": true
}
```

## Quality Checks

Before marking as complete:

- [ ] All 200 examples reviewed
- [ ] All `confidence` values are 1.0
- [ ] All `is_gold` values are true
- [ ] Segmentation is linguistically correct
- [ ] No JSON formatting errors

## Resources

### Online Tools
- [Sanskrit Dictionary](https://www.sanskrit-lexicon.uni-koeln.de/)
- [DCS (Digital Corpus of Sanskrit)](http://www.sanskrit-linguistics.org/dcs/)
- [Gita Supersite](http://www.gitasupersite.iitk.ac.in/) - has pada-patha

### Books
- "A Sanskrit Grammar for Students" - Arthur A. Macdonell
- "Sanskrit Sandhi and Exercises" - M. M. Ghare

## Tips

1. **Use pada-patha references**: Many Gita editions show word-by-word splits
2. **Check multiple sources**: If unsure, consult 2-3 references
3. **Mark difficult cases**: Add a comment if you're uncertain
4. **Take breaks**: Don't annotate more than 50 verses at a time
5. **Double-check compounds**: These are the trickiest

## Contact

If you encounter difficult cases or need clarification, please:
1. Mark the example with a comment
2. Continue with other examples
3. Discuss unclear cases before finalizing

## Completion

Once all 200 examples are annotated:
1. Verify JSON format: `python -m json.tool data/benchmarks/sandhi_gold_test.jsonl`
2. Run validation: `python scripts/validate_gold_annotations.py`
3. Commit the annotated file
