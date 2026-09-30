from PIL import Image, ImageDraw, ImageFont

img = Image.new('RGB', (1400, 1000), 'white')
d = ImageDraw.Draw(img)

# Fonts
try:
    title_font = ImageFont.truetype('arial.ttf', 28)
    box_font = ImageFont.truetype('arial.ttf', 18)
except Exception:
    title_font = ImageFont.load_default()
    box_font = ImageFont.load_default()

# Colors
blue = '#EAF2FF'
edge = '#2E5BBA'
line = '#3E4A59'
text = '#111111'

# Boxes: (name, x, y, w, h, label)
boxes = [
    ('sat', 60, 120, 260, 140, 'Satellite Data\nCyclone_Images.h5'),
    ('ibtracs', 360, 120, 260, 140, 'IBTrACS\nStorm Tracks'),
    ('nwp', 660, 120, 260, 140, 'NWP / ERA5\nWeather Grids'),
    ('prep1', 60, 350, 260, 150, 'process_cyclone_images.py\nLoad + normalize'),
    ('prep2', 360, 350, 260, 150, 'process_ibtracs.py\nClean + categorize'),
    ('prep3', 660, 350, 260, 150, 'fetch_nwp_data.py\nFetch / generate'),
    ('fusion', 350, 620, 430, 170, 'data_fusion_pipeline.py\nFuse image + NWP + track data'),
    ('split', 350, 820, 430, 120, 'train_val_test_split.py\nTrain / Val / Test split'),
    ('m1', 80, 900, 280, 100, 'Model 1\nVortex eye detection\n& segmentation'),
    ('m2', 430, 900, 280, 100, 'Model 2\nTrack forecasting'),
    ('pred', 780, 900, 280, 100, 'predict_track.py\nInference + map output'),
]

# Draw title

d.text((420, 30), 'CycloneVision AI Project Architecture', fill=text, font=title_font)

# Draw boxes
for _, x, y, w, h, label in boxes:
    d.rounded_rectangle((x, y, x + w, y + h), radius=15, fill=blue, outline=edge, width=2)
    d.multiline_text((x + 18, y + 22), label, fill=text, font=box_font, spacing=6)

# Draw arrows between stages
arrow_pairs = [
    ((190, 260), (190, 350)),
    ((490, 260), (490, 350)),
    ((790, 260), (790, 350)),
    ((190, 500), (405, 620)),
    ((490, 500), (560, 620)),
    ((790, 500), (560, 620)),
    ((560, 790), (560, 820)),
    ((560, 940), (560, 900)),
    ((220, 1000), (220, 1000)),
]

for (x1, y1), (x2, y2) in arrow_pairs:
    d.line((x1, y1, x2, y2), fill=line, width=2)
    d.polygon([(x2, y2), (x2 - 10, y2 - 10), (x2 + 10, y2 - 10)], fill=line)

# More direct arrows for the bottom row
for x in [220, 570, 920]:
    d.line((x, 820, x, 900), fill=line, width=2)
    d.polygon([(x, 900), (x - 10, 890), (x + 10, 890)], fill=line)

output_path = r'c:\Users\HP\Desktop\cyclone\project_architecture.png'
img.save(output_path)
print(f'Saved: {output_path}')
