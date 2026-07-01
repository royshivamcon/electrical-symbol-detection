# Electrical EDA — Results

Summary of findings from `../../electrical_data_testing.ipynb`. All numbers are from the cached notebook outputs.

Source CSV: `../../data/Lighting and Power_Electrical - duplication_state_worksheets_electrical.csv`

---

## 1. Corpus

| metric              | value |
|---------------------|------:|
| worksheets (rows)   | 3,206 |
| source requests     |    46 |

### Worksheets per request

| metric | value |
|--------|------:|
| mean   |  69.7 |
| min    |    13 |
| max    |   429 |
| 95th % | 186.0 |

---

## 2. Cached geometry coverage

After `fetcher.fetch_all_request_data(... fetch_geometries=True)`:

| metric                       | value         |
|------------------------------|--------------:|
| worksheets with cached JSON  | 1,924 / 3,206 |
| total feature layers         |        34,172 |

---

## 3. Geometry types

| type        |  count |
|-------------|-------:|
| Point       | 31,292 |
| LineString  |  1,155 |

The dataset is overwhelmingly point-symbol data. LineStrings show up but Polygons are absent from the cached set.

---

## 4. Point features

| metric              | value   |
|---------------------|--------:|
| total point features|218,182  |
| worksheets w/ points|  1,772  |

### Points per worksheet

| metric | value |
|--------|------:|
| count  | 1,772 |
| mean   | 123.1 |
| min    |     1 |
| max    | 2,036 |
| 95th % | 481.0 |
| std    | 167.1 |

Long-tailed distribution: a few worksheets hold thousands of points; the typical sheet is ~120.

---

## 5. Feature-name distribution (per request)

For each request, count unique feature names whose occurrence is at or above that request's 95th percentile:

| metric | value |
|--------|------:|
| mean   |  12.2 |
| min    |     5 |
| max    |    46 |
| 95th % |  22.8 |

So a typical request has ~12 dominant feature names. The most diverse one has 46.

---

## 6. Sample worksheet rendering

Picked one worksheet (`refs[0]`), rendered the PDF page at 2× scale via `fitz.Matrix(2, 2)`:

| metric                   | value           |
|--------------------------|-----------------|
| Feathers FE dimensions   | 7000 × 4952 px  |
| Rendered pixmap          | 6740 × 4768 px  |
| Saved as                 | `../images/img1.png` |

`../images/img1.png` is the input image to the table-processing experiments.

---