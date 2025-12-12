# Raw Data Directory

This directory contains raw Sanskrit text files before normalization.

## Expected Format

### Plain Text Format (`.txt`)
One verse per line with verse ID:

```
1.1: धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।
1.2: सञ्जय उवाच। दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा।
```

### JSON Format (`.json`)
```json
{
  "verses": [
    {
      "id": "1.1",
      "text": "धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।"
    },
    {
      "id": "1.2",
      "text": "सञ्जय उवाच। दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा।"
    }
  ]
}
```

### JSONL Format (`.jsonl`)
```jsonl
{"id": "1.1", "text": "धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।"}
{"id": "1.2", "text": "सञ्जय उवाच। दृष्ट्वा तु पाण्डवानीकं व्यूढं दुर्योधनस्तदा।"}
```

## Sources

Recommended sources for Bhagavad Gita text:
- [Sanskrit Documents](https://sanskritdocuments.org/)
- [Gitasupersite](http://www.gitasupersite.iitk.ac.in/)
- [Archive.org Sanskrit texts](https://archive.org/)

## Note

This directory is in `.gitignore` to avoid committing large text files.
Download your source data and place it here as `gita_raw.txt` or similar.
