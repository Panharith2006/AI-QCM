# Report Generation Guide

## Quick Start

Generate comprehensive before/after reports for your OMR pipeline:

```bash
python scripts/generate_reports.py --image path/to/test_sheet.jpg
```

This creates:
- Before/after comparison images for each layer
- Detailed markdown explanations
- Performance metrics and timings
- All saved to `reports/` directory

## Usage

### Basic Usage
```bash
python scripts/generate_reports.py --image sample_sheet.jpg
```

### Custom Output Directory
```bash
python scripts/generate_reports.py --image sample_sheet.jpg --output my_reports
```

### Custom YOLO Model
```bash
python scripts/generate_reports.py --image sample_sheet.jpg --yolo path/to/custom_model.pt
```

### All Options
```
--image     (required)  Path to test sheet image
--yolo      (optional)  Path to YOLO model (default: artifacts/yolo/best.pt)
--output    (optional)  Output directory for reports (default: reports)
```

## What Gets Generated

### Comparison Images
- `layer_2_preprocessing/comparison/layer_2_clahe_comparison.jpg` - Before/after CLAHE
- `layer_3_alignment/comparison/layer_3_perspective_comparison.jpg` - Before/after perspective correction
- `layer_4_yolo_detection/comparison/layer_4_yolo_detections.jpg` - Detection visualization
- `layer_5_6_extraction/comparison/layer_5_6_mcq_crop_example.jpg` - Example MCQ crop

### Markdown Reports
- `layer_1_input_acquisition/explanation.md` - Image loading & validation
- `layer_2_preprocessing/explanation.md` - CLAHE algorithm details & impact
- `layer_3_alignment/explanation.md` - Edge detection & perspective transform
- `layer_4_yolo_detection/explanation.md` - YOLO architecture & detections
- `layer_5_6_extraction/explanation.md` - Extraction algorithms & fill ratios
- `layer_7_mapping/explanation.md` - Answer mapping structure
- `scoring_optional/explanation.md` - Scoring & comparison metrics
- `technology_stack/overview.md` - Technology choices & alternatives
- `end_to_end_pipeline/pipeline_overview.md` - Complete pipeline flow

### Analysis Templates
Each layer folder contains an `analysis_template.md` for your custom notes:
- Observations and findings
- Parameter tuning experiments
- Edge cases discovered
- Recommendations

## Working with Generated Reports

### 1. Review Auto-Generated Content
Open each `explanation.md` to understand:
- What each layer does
- Why this approach was chosen
- How parameters affect results
- Performance statistics

### 2. Add Your Own Analysis
Fill in the `analysis_template.md` files with:
- Results from your tests
- Parameter tuning experiments
- Performance measurements
- Problem cases encountered

### 3. Examine Comparison Images
Look at side-by-side before/after images to see:
- Effect of CLAHE preprocessing
- Quality of perspective correction
- Detection accuracy
- Example extracted regions

### 4. Generate Multiple Test Sets
Run the script on different types of sheets:
- Well-scanned sheets (baseline)
- Poorly scanned sheets (stress test)
- Different mark types (pencil, pen, highlighter)
- Different question formats (Circle Fill, Roman Numeral, TFNG, Alpha Box)

```bash
python scripts/generate_reports.py --image good_scan.jpg --output reports/good_scans
python scripts/generate_reports.py --image poor_scan.jpg --output reports/poor_scans
```

### 5. Compare Results
Keep multiple report sets to document:
- Improvements after bug fixes
- Performance across different test sets
- Algorithm changes and their impact

## Example Report Usage

After running the script, your report structure looks like:

```
reports/
├── README.md
├── layer_1_input_acquisition/
│   └── explanation.md
├── layer_2_preprocessing/
│   ├── explanation.md                    # Auto-generated
│   ├── comparison/
│   │   ├── layer_2_clahe_comparison.jpg  # Image showing CLAHE effect
│   │   └── analysis_template.md          # Your observations go here
├── layer_3_alignment/
│   ├── explanation.md                    # Auto-generated
│   └── comparison/
│       ├── layer_3_perspective_comparison.jpg
│       └── analysis_template.md          # Your observations
...
```

## Tips for Effective Reports

1. **Test Multiple Sheet Types**
   - Generate reports for different sheet qualities
   - Document which types work best/worst

2. **Document Parameter Tuning**
   - Test different CLAHE clip_limit values
   - Try different Canny threshold combinations
   - Record accuracy changes

3. **Include Your Findings**
   - Add custom analysis to template files
   - Explain why certain parameters work better
   - Document edge cases you discover

4. **Track Improvements**
   - Generate reports after each major fix
   - Compare accuracy metrics
   - Document performance gains

5. **Use for Documentation**
   - Include comparison images in your final report
   - Reference the generated explanations
   - Show before/after results to stakeholders

## Troubleshooting

### "Model not found"
```
⚠ Warning: YOLO model not found, skipping detection
```
Make sure the YOLO model exists at the path specified (default: `artifacts/yolo/best.pt`)

### "Could not load image"
Check that:
- Image file path is correct
- Image format is supported (JPG, PNG, BMP, TIFF)
- Image file is not corrupted

### Empty or black images
If comparison images are all black:
- Check image dimensions are reasonable
- Verify image is properly loaded
- Check color space (BGR vs RGB)

## Next Steps

1. Run the script on your test images
2. Review generated reports and images
3. Fill in analysis templates with your observations
4. Use reports in your final project documentation
5. Share with team members or instructors
