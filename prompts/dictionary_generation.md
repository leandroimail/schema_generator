# Data Dictionary Generation

## Objective
Act as a documentation and data modeling expert. Generate a complete data dictionary from profiling artifacts and a data sample.

---

## Instructions

1. **Context**  
   - Receive as input two blocks:  
     - `## Data Profile` – profiling statistics (types, counts, nulls, distribution, etc.).  
     - `## Data Sample` – a sample of real records.

2. **Table Description**  
   - Infer the table name (`table_name`) from the profile metadata when available.
   - Generate a general description of the table's application / utility (`table_description`).

3. **Field Description**  
   Generate exactly one object for every column present in the profile/sample, using these fields:
   - `field_name` (string)  
   - `data_type` (string) – inferred data type  
   - `field_description` (string) – purpose and relation to the rest of the model  
   - `example_value` (string, number, boolean, or null) – representative scalar value extracted from the sample  
   - `domain_values` (array, optional) – if it is an enum or restricted domain, list all known values. Omit this field when it is not applicable.  
   - `full_description` (string) – concatenate `field_description` with either `Domain: <domain_values>` when `domain_values` exists, or `Example: <example_value>` otherwise.

4. **Output Format**  
   - Return **only** one valid JSON object, without Markdown, comments, explanations, or surrounding text.  
   - The JSON object must follow this structure:

     ```json
     {
       "table_name": "table_name",
       "table_description": "Brief description and use of the table",
       "fields": [
         {
           "field_name": "column1",
           "data_type": "string|integer|datetime|...",
           "field_description": "Detailed description",
           "example_value": "...",
           "domain_values": [ "A", "B", "C" ],
           "full_description": "Detailed description - Domain: [ \"A\", \"B\", \"C\" ]"
         },
         {
           "field_name": "column2",
           "data_type": "...",
           "field_description": "Detailed description",
           "example_value": 123,
           "full_description": "Detailed description - Example: 123"
         }
       ]
     }
     ```

---
## Data Profile

```json
<profile>
```

---
## Data Sample

```csv
<sample>
```
