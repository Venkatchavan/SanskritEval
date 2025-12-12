# Sanskrit Data Sources for Bhagavad Gita

## Current Status
- ✅ **gita_raw.txt**: 10 verses (Chapters 1-4)
- ⚠️  Other files are English translations

## Where to Get Sanskrit Verses

### 1. Gita Supersite (IIT Kanpur) - **RECOMMENDED**
- URL: http://www.gitasupersite.iitk.ac.in/
- **Has**: All 700 verses in Devanagari with pada-patha (word splits!)
- **License**: Academic use (verify)
- **Format**: Can copy verse by verse or use their API

### 2. Sanskrit Documents Archive
- URL: https://sanskritdocuments.org/
- **Has**: Multiple Gita versions in Devanagari
- **License**: Public domain
- **Format**: Plain text files

### 3. DCS (Digital Corpus of Sanskrit)
- URL: http://www.sanskrit-linguistics.org/dcs/
- **Has**: Gita with morphological annotations
- **License**: Academic
- **Format**: XML/JSON

### 4. Archive.org
- URL: https://archive.org/
- Search: "Bhagavad Gita Sanskrit"
- **Has**: Scanned books, some with OCR
- **License**: Varies, many public domain
- **Format**: PDF/text

### 5. GitHub Repositories
- Search: "bhagavad gita sanskrit devanagari"
- Many community projects have verse collections
- **License**: Check each repo
- **Format**: Usually JSON/text

## Recommended Approach

### Option A: Quick (for initial testing)
1. Manually copy verses from Gita Supersite
2. Create `gita_full.txt` with all 700 verses
3. Format: `chapter.verse: text`

### Option B: Comprehensive (best for research)
1. Use Gita Supersite data (has pada-patha!)
2. Their word splits = **gold labels** for sandhi task
3. Can directly compare model outputs to expert splits

### Option C: Automated (if API available)
1. Check if any source has API/bulk download
2. Script the data collection
3. Automate format conversion

## Data Format We Need

```
1.1: धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः। मामकाः पाण्डवाश्चैव किमकुर्वत सञ्जय।।
1.2: सञ्जय उवाच। दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा। आचार्यमुपसङ्गम्य राजा वचनमब्रवीत्।।
...
18.78: यत्र योगेश्वरः कृष्णो यत्र पार्थो धनुर्धरः। तत्र श्रीर्विजयो भूतिर्ध्रुवा नीतिर्मतिर्मम।।
```

## Gita Supersite - Best Source

### Why?
1. ✅ **All 700 verses** in Devanagari
2. ✅ **Pada-patha available** (word-by-word splits)
3. ✅ **Multiple commentaries** (for context)
4. ✅ **Well-maintained** by IIT Kanpur
5. ✅ **Widely cited** in academic work

### What to Download
- **Sandhi form** (samhita-patha): For input
- **Pada-patha** (split form): For gold labels!
- This gives us **perfect segmentation** without manual annotation

### Example from Gita Supersite
```
Verse 1.1:
Samhita: धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः
Pada:    धर्मक्षेत्रे | कुरुक्षेत्रे | समवेताः | युयुत्सवः

This is GOLD DATA! 🎯
```

## Action Items

### Immediate (to expand dataset)
1. [ ] Visit Gita Supersite
2. [ ] Extract verses in Devanagari (samhita form)
3. [ ] Extract pada-patha (split form) if available
4. [ ] Create `data/raw/gita_full_700.txt`
5. [ ] Regenerate datasets with `--silver-size 700`

### Better (for gold labels)
1. [ ] Parse pada-patha from Gita Supersite
2. [ ] Use as gold labels directly
3. [ ] Skip manual annotation for 200 verses
4. [ ] Have entire dataset as gold standard!

## Quick Commands

```bash
# After getting full data
python scripts/generate_sandhi_data.py \
    --input data/raw/gita_full_700.txt \
    --gold-size 200 \
    --silver-size 700

# Or if we get pada-patha
python scripts/convert_padapatha_to_gold.py \
    --input data/raw/gita_padapatha.txt \
    --output data/benchmarks/sandhi_gold_full.jsonl
```

## License Considerations

- ✅ Gita itself: Public domain (ancient text)
- ⚠️  Specific editions/translations: May have copyright
- ⚠️  Digital versions: Check terms of use
- ✅ For academic research: Usually permitted with citation
- ✅ For benchmark datasets: Cite source appropriately

## Citation

If using Gita Supersite data:
```
@misc{gitasupersite,
  title={Bhagavad Gita},
  author={{IIT Kanpur}},
  year={2023},
  url={http://www.gitasupersite.iitk.ac.in/}
}
```

## Next Steps

**Recommended**: 
1. Get Gita Supersite data (700 verses + pada-patha)
2. This gives us both samhita and pada forms
3. Pada-patha = perfect gold labels!
4. No need for manual annotation

This would make the benchmark much stronger! 🚀
