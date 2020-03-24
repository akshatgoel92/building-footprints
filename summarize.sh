export PYTHONPATH=.

python summarize/get_masks.py --region=Deoria --invert=False
python summarize/get_masks.py --region=Deoria --invert=True
python summarize/get_masks.py --region=Gorakhpur --invert=False
python summarize/get_masks.py --region=Gorakhpur --invert=True


